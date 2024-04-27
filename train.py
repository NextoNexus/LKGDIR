# python imports
import os
import glob
import random
import warnings
import sys
from argparse import ArgumentParser

# external imports
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import nibabel as nib
from matplotlib import pyplot as plt
from pathlib import Path
import pickle

# internal imports
from model import SpatialTransform, LKG_Net
from losses import NCC, mse, smoothloss
from test import test
from reusable import make_exp_dir, save_args, save_metrics, brain_preProcess
import Unaligned_Datasets



def train(dir_path,
          device,
          atlas_dir,
          lr,
          n_epoch,
          data_loss,
          model_type,
          start_channel,
          remain_channels,
          reg_param,
          batch_size,
          reduc_size,
          slurm_data_path,
          parallel_sizes):
    """
    model training function
    :param dir_path: current trial dir
    :param gpu: integer specifying the gpu to use
    :param atlas_dir: atlas dirname.
    :param lr: learning rate
    :param n_epoch: number of training epochs
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model_type: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param reduc_size: whether reduce image size
    :param slurm_data_path: path to all of the data
    :param parallel_sizes: kernel sizes of parallel convs
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ########################################################
    # brain_affine.nii.gz = array([Sagittal, coronal, axial])
    ########################################################

    #atlas_vol=nib.load(atlas_dir+'/'+'brain_affine.nii.gz').get_fdata()
    #atlas_vol=brain_preProcess(atlas_vol)

    atlas_vol = nib.load(atlas_dir + '/' + 'vol_atlas.nii.gz').get_fdata()
    atlas_seg = nib.load(atlas_dir + '/' + 'seg_atlas.nii.gz').get_fdata()
    atlas_tensor=brain_preProcess(atlas_vol,reduce_size=reduc_size).to(device)

    B, C, D, H, W = atlas_tensor.shape

    vol_size = atlas_tensor.shape[2:5]

    # create training dataset and dataloader
    train_datasets=Unaligned_Datasets.IxIDataset_for_atlas2img(slurm_data_path=slurm_data_path,cases_dir='trainset',need_seg=False,reduce_size=reduc_size)
    train_dataloader=DataLoader(train_datasets,batch_size=batch_size,shuffle=True)

    # Prepare the vm1 or vm2 model and send to device
    remain_channels = eval(remain_channels)
    parallel_sizes = eval(parallel_sizes)
    model = LKG_Net(
        vol_shape=vol_size,
        parallel_sizes=parallel_sizes,
        start_channel=start_channel,
        remain_channels=remain_channels
    )
    print('LKG-Net used')

    # compute size of model
    params = list(model.named_parameters())
    unique_params = {}
    for name, param in params:
        if param.requires_grad and name not in unique_params:
            unique_params[name] = param.numel()
    num_params = sum(unique_params.values())
    print('size of model:',num_params)
    print('#########################')

    model.to(device)

    # Set optimizer and losses
    optimizer = Adam(model.parameters(), lr=lr)
    sim_loss_fn = NCC() if data_loss == "ncc" else mse
    grad_loss_fn = smoothloss
    
    transform = SpatialTransform(device=device).to(device)
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    Loss=[]
    Sim_loss=[]
    Grad_loss=[]
    ValDice=[]
    ValJac=[]
    torch.cuda.empty_cache()

    # Training loop.
    for epoch in range(1, n_epoch+1):
        print('\n======================>Epoch {} start'.format(epoch))
        temp_loss=0
        temp_grad_loss=0
        temp_recon_loss=0
        train_dataloader_iter = iter(train_dataloader)
        model.train()
        for i in range(train_datasets.__len__()):
            print('---------------->Epoch : {}, Iteration : {}'.format(epoch,i))
            fixed_image=next(train_dataloader_iter)

            fixed_tensor = fixed_image.to(device)

            flow = model(atlas_tensor, fixed_tensor)
            warpped_tensor = transform(atlas_tensor, flow.permute(0, 2, 3, 4, 1))
            # warp = trf(input_moving,flow)

            # Calculate loss
            recon_loss = sim_loss_fn(fixed_tensor, warpped_tensor)
            grad_loss = grad_loss_fn(flow)
            loss = recon_loss + reg_param * grad_loss

            print('Sim loss : {}------Grad loss : {}------Total loss : {}'.format(recon_loss,grad_loss,loss))

            # Backwards and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            temp_recon_loss = temp_recon_loss+ recon_loss.item()
            temp_grad_loss = temp_grad_loss+ grad_loss.item()
            temp_loss = temp_loss+ loss.item()

        print('\n---------------->Iteration end')

        temp_loss=temp_loss/train_datasets.__len__()
        temp_grad_loss=temp_grad_loss/train_datasets.__len__()
        temp_recon_loss=temp_recon_loss/train_datasets.__len__()

        save_metrics(temp_loss, dir_path, "Total Loss.txt")
        save_metrics(temp_grad_loss, dir_path, "Smooth Loss.txt")
        save_metrics(temp_recon_loss, dir_path, "Sim Loss.txt")

        print('Average Sim loss : {}------Average Grad loss : {}------Average Total loss : {}'.format(temp_recon_loss,temp_grad_loss,temp_loss))

        Sim_loss.append(temp_recon_loss)
        Grad_loss.append(temp_grad_loss)
        Loss.append(temp_loss)

        if epoch % 10 == 0 or epoch == n_epoch:
            # evaluation
            print('\n----------->evaluation')
            model.eval()
            mean_dice_for_brainRegion,mean_dice_for_subject,neg_jac_fraction_list = test(dir_path=dir_path,
                                                                                         device=device,
                                                                                         atlas_dir=atlas_dir,
                                                                                         model_load=model,
                                                                                         current_epoch=epoch,
                                                                                         reduc_size=reduc_size,
                                                                                         slurm_data_path=slurm_data_path
                                                                                        )
            ValDice.append(np.mean(mean_dice_for_brainRegion))
            ValJac.append(sum(neg_jac_fraction_list)/len(neg_jac_fraction_list))

            save_metrics(sum(neg_jac_fraction_list) / len(neg_jac_fraction_list), dir_path, "mean_neg_jac_fraction.txt")
            save_metrics(mean_dice_for_brainRegion, dir_path, "mean_dice_for_brainRegion.txt")
            save_metrics(mean_dice_for_subject, dir_path, "mean_dice_for_subject.txt")
            save_metrics(np.mean(mean_dice_for_brainRegion), dir_path, "mean_dice.txt")


            # print('Validation Dice score:',validation_avg_dice)
            print('Average of mean_dice_for_brainRegion :', np.mean(mean_dice_for_brainRegion))
            print('Average of mean_dice_for_subject :', np.mean(mean_dice_for_subject))

            # save checkpoint
            checkpoint_dir=os.path.join(dir_path,'checkpoint')
            if not os.path.exists(checkpoint_dir):
                Path(checkpoint_dir).mkdir()
            model_path = os.path.join(checkpoint_dir,'TrainLoss_{}_Epoch_{}_checkpoint'.format(np.round(temp_loss,6),epoch))
            torch.save({
                'model_type' : model_type,
                'start_channel':start_channel,
                'remain_channels':remain_channels,
                'num_of_parameters': num_params,
                'parallel_sizes' : parallel_sizes,
                'epoch': epoch,
                'resolution':'half' if reduc_size else 'full',
                'model_state_dict': model.state_dict(),
                'loss': temp_loss}, model_path)

            # visualize
            #x = np.arange(1, epoch+1)
            fig=plt.figure()
            plt.title('loss curve')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(Loss, label='total loss')
            plt.plot(Sim_loss, label='Sim loss')
            plt.legend()
            fig.savefig(os.path.join(dir_path, '{}_Loss_curve_epoch_{}.jpg'.format(model_type, epoch)))
            #plt.show()






if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--device", type=str, default='cuda', dest='device', choices=['cpu', 'cuda'])
    parser.add_argument("--atlas_dir",type=str,dest="atlas_dir",default='../../../data/atlas')
    parser.add_argument("--lr", type=float,default=1e-4, dest="lr", help="learning rate")
    parser.add_argument("--n_epoch",type=int, default=100, dest="n_epoch", help="number of epochs")
    parser.add_argument("--data_loss",type=str,dest="data_loss",default='ncc',help="data_loss: mse of ncc")
    parser.add_argument("--model_type",type=str,dest="model_type",choices=['LKG-Net'],default='LKG-Net')
    parser.add_argument("--start_channel", type=int, dest="start_channel", default=16)
    parser.add_argument("--remain_channels", type=str, dest="remain_channels", default='[32,16,16]')
    parser.add_argument("--lambda",type=float,default=4.0, dest="reg_param",help="regularization parameter")
    parser.add_argument("--batch_size",type=int,dest="batch_size",default=1,help="batch_size")
    parser.add_argument("--reduce",type=int, default=1, dest="reduc_size",help="Reduce size of volumes")
    parser.add_argument("--slurm_data_path",type=str,default='../../../data', dest="slurm_data_path")

    # set multi-resolution conv
    parser.add_argument("--parallel_sizes", type=str, default='[1,3,7]', help="set multi-resolution conv for LK_DMD")
    args = parser.parse_args()

    # Make Trial Directory
    par_dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = make_exp_dir(par_dir_path)

    save_args(dir_path,args)
    
    train(dir_path,**vars(parser.parse_args()))





