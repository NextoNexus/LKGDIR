# python imports
import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import math
import nibabel as nib
from model import SpatialTransform, LKG_Net
import Unaligned_Datasets
from reusable import plot_brain_and_save, dice, brain_preProcess, seg_preProcess, get_neg_jac_fraction



def test(dir_path, device, atlas_dir, model_load, current_epoch, reduc_size, slurm_data_path):
    """
    model training function
    :param dir_path: trial dir
    :param gpu: integer specifying the gpu to use
    :param atlas_dir: atlas dirname.
    :param model_load: load actual model, from training or trained model
    :param current_epoch: whether to reduce size of volumes
    :param reduc_size: whether reduce resolution
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param slurm_data_path: path to all of the data

    """

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    val_datasets = Unaligned_Datasets.IxIDataset_for_atlas2img(slurm_data_path=slurm_data_path, cases_dir='valset',need_seg=True,reduce_size=reduc_size)
    val_dataloader = DataLoader(val_datasets, batch_size=1, shuffle=False)

    good_labels=[2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
         
    # set up fixed image
    ########################################################
    # brain_affine.nii.gz = array([Sagittal, coronal, axial])
    ########################################################
    #atlas_vol = nib.load(atlas_dir + '/' + 'brain_affine.nii.gz').get_fdata()
    # brain preProcess
    #atlas_vol = brain_preProcess(atlas_vol)
    atlas_vol = nib.load(atlas_dir + '/' + 'vol_atlas.nii.gz').get_fdata()
    atlas_seg = nib.load(atlas_dir + '/' + 'seg_atlas.nii.gz').get_fdata()
    atlas_tensor = brain_preProcess(atlas_vol, reduce_size=reduc_size).to(device)
    atlas_seg_tensor = seg_preProcess(atlas_seg, reduce_size=reduc_size).to(device)

    B,C,D,H,W=atlas_tensor.shape
    vol_size = atlas_tensor.shape[2:5]

    # set up fixed segmentation
    #fixed_seg_arr = nib.load(atlas_dir + '/' + 'aseg_affine.nii.gz').get_fdata()
    # seg preProcess
    #fixed_seg_arr = seg_preProcess(fixed_seg_arr)

    # Use this to warp segments
    transform = SpatialTransform(device=device).to(device)

    #load model
    if isinstance(model_load, str):
        checkpoint_path = os.path.join(dir_path, model_load)
        checkpoint = torch.load(checkpoint_path)

        current_epoch = checkpoint['epoch']
        parallel_sizes = checkpoint['parallel_sizes']
        model_type = checkpoint['model_type']
        start_channel = checkpoint['start_channel']
        remain_channels = checkpoint['remain_channels']
        num_params=checkpoint['num_of_parameters']

        print('################################')
        print('current_epoch :',current_epoch)
        print('parallel_sizes :',parallel_sizes)
        print('model_type:',model_type)
        print('start_channel:',start_channel)
        print('remain_channels:',remain_channels)
        print('num_params:',num_params)
        print('################################')


        model = LKG_Net(
            vol_shape=vol_size,
            parallel_sizes=parallel_sizes,
            start_channel=start_channel,
            remain_channels=remain_channels
        )
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        model = model_load

    val_dataloader_iter = iter(val_dataloader)
    dice_array = np.zeros((1,val_datasets.__len__()))
    neg_jac_fraction_list=[]

    # torch.cuda.empty_cache()
    #  evaluation loop
    with torch.no_grad():
        for k in range(0, val_datasets.__len__()):
            model.eval()
            fixed_image, fixed_seg=next(val_dataloader_iter)

            fixed_tensor = fixed_image.to(device)

            '''print('type of atlas_vol:', type(atlas_vol[0][0][0][0][0]))
            print('type of fixed_img:', type(fixed_image.numpy()[0][0][0][0][0]))
            print('min of atlas_vol:', np.min(atlas_vol), 'max of atlas_vol:',
                  np.max(atlas_vol))
            print('min of fixed_image:', np.min(fixed_image.numpy()), 'max of fixed_image:',
                  np.max(fixed_image.numpy()))'''

            flow = model(atlas_tensor, fixed_tensor)


            neg_jac_fraction = get_neg_jac_fraction(device,flow)
            wrapped_tensor=transform(atlas_tensor, flow.permute(0, 2, 3, 4, 1))
            wrapped_arr = wrapped_tensor.detach().cpu().numpy()

            # visualize
            plot_brain_and_save(fixed_image.numpy(), atlas_vol, wrapped_arr, dir_path, k, current_epoch)

            # compute wrapped segmentation
            warpped_seg_arr = transform(atlas_seg_tensor, flow.permute(0, 2, 3, 4, 1), mod = 'nearest').detach().cpu().numpy()

            '''print('type of warpped_seg_arr:',type(warpped_seg_arr[0][0][0][0][0]))
            print('type of fixed_seg:', type(fixed_seg.numpy()[0][0][0][0][0]))
            print('min of warpped_seg_arr:',np.min(warpped_seg_arr),'max of warpped_seg_arr:',np.max(warpped_seg_arr))
            print('min of fixed_seg:', np.min(fixed_seg.numpy()), 'max of fixed_seg:',
                  np.max(fixed_seg.numpy()))'''

            vals = dice(warpped_seg_arr, fixed_seg.numpy(), labels=good_labels, nargout=1)

            dices = vals

            '''
            Idx   SegId   SegName                                       Idx  SegId    SegName
             0    2       Left-Cerebral-White-Matter                    14   24       CSF
             1    3       Left-Cerebral-Cortex                          15   28       Left-VentralDC
             2    4       Left-Lateral-Ventricle                        16   41       Right-Cerebral-White-Matter
             3    7       Left-Cerebellum-White-Matter                  17   42       Right-Cerebral-Cortex
             4    8       Left-Cerebellum-Cortex                        18   43       Right-Lateral-Ventricle
             5   10       Left-Thalamus-Proper                          19   46       Right-Cerebellum-White-Matter
             6   11       Left-Caudate                                  20   47       Right-Cerebellum-Cortex
             7   12       Left-Putamen                                  21   49       Right-Thalamus-Proper
             8   13       Left-Pallidum                                 22   50       Right-Caudate
             9   14       3rd-Ventricle                                 23   51       Right-Putamen
            10   15       4th-Ventricle                                 24   52       Right-Pallidum
            11   16       Brain-Stem                                    25   53       Right-Hippocampus
            12   17       Left-Hippocampus                              26   54       Right-Amygdala
            13   18       Left-Amygdala                                 27   60       Right-VentralDC

            segsNames = ['Cerebral-White-Matter', 'Cerebral-Cortex', 'Lateral-Ventricle', 'Cerebellum-White-Matter',
                         'Cerebellum-Cortex', 'Thalamus-Proper', 'Caudate', 'Putamen', 'Pallidum', '3rd-Ventricle',
                         '4th-Ventricle', 'Brain-Stem', 'Hippocampus', 'Amygdala', 'CSF', 'VentralDC']
            '''
            vals_all = [(dices[0] + dices[16]) / 2, (dices[1] + dices[17]) / 2, (dices[2] + dices[18]) / 2,
                        (dices[3] + dices[19]) / 2,
                        (dices[4] + dices[20]) / 2, (dices[5] + dices[21]) / 2, (dices[6] + dices[22]) / 2,
                        (dices[7] + dices[23]) / 2,
                        (dices[8] + dices[24]) / 2, dices[9], dices[10], dices[11],
                        (dices[12] + dices[25]) / 2, (dices[13] + dices[26]) / 2, dices[14], (dices[15] + dices[27]) / 2]

            vals_all = [round(i, 9) for i in vals_all]
            print('eval done. the Dice for {}th image is :{}----------the neg_jac_fraction is {}'.format(k+1,np.mean(vals_all),neg_jac_fraction))

            if k==0:
                dice_array = vals_all
            else:
                dice_array  = np.vstack((dice_array,vals_all))
            neg_jac_fraction_list.append(neg_jac_fraction.item())

    mean_dice_for_brainRegion = np.mean(dice_array,axis=0)
    mean_dice_for_subject=np.mean(dice_array,axis=1)
    return mean_dice_for_brainRegion, mean_dice_for_subject, neg_jac_fraction_list



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir_path", type=str, default='./Trials/Trial_2', help="trial dir")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--atlas_dir",type=str,dest="atlas_dir",default='../../../data/atlas')
    parser.add_argument("--model_load",type=str,dest="model_load",help="Model path to be loaded", default=r'checkpoint/TrainLoss_-0.316647_Epoch_100_checkpoint')
    parser.add_argument("--current_epoch",type=int,dest="current_epoch",help="current epoch of trained model", default=None)
    parser.add_argument("--reduce", type=int, default=1, dest="reduc_size", help="Reduce size of volumes")
    #parser.add_argument("--test_file",type=str,dest="test_file",default='../../src/test_volnames.txt',help="Test Text File")
    parser.add_argument("--slurm_data_path",type=str,dest="slurm_data_path", default='../../../data')

    mean_dice_for_brainRegion, mean_dice_for_subject, neg_jac_fraction_list =  test(**vars(parser.parse_args()))
    print(mean_dice_for_brainRegion)
    print(mean_dice_for_subject)
    print(sum(neg_jac_fraction_list)/len(neg_jac_fraction_list))

