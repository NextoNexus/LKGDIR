import os
import json
from pathlib import Path
import torch
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as nnf
from plugin import Poisson
import time

def make_exp_dir(par_dir_path):
    i = 1
    if not os.path.exists(str(par_dir_path) + "/Trials/Trial_%s" % i):
        pathname = str(par_dir_path) + "/Trials/Trial_" + str(i)
    else:
        while os.path.exists(str(par_dir_path) + "/Trials/Trial_%s" % i):
            i += 1
        pathname = str(par_dir_path) + "/Trials/Trial_" + str(i)
    
    Path(pathname).mkdir()
    return pathname
      
    
def save_args(dir_path,args):
    filename = dir_path + '/Arguments_Hyperparamaters.txt'
    args.__dict__['create time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
        
def save_metrics(metric, dir_path, metric_name):
    
    if torch.is_tensor(metric):
        metric = metric.detach().cpu().numpy()
    
    file = dir_path + "/" + metric_name
    if not os.path.exists(file):
        f = open(file, "x")
    f = open(file, "a")
    f.write("%s\n" % metric)
    f.close()

def plot_brain_and_save(input_fixed_arr, moving_image, wrap_image, dir_path, case_idx, current_epoch):
    # visualize
    # sagittal
    input_fixed_arr = input_fixed_arr.squeeze()
    moving_image = moving_image.squeeze()
    wrap_image = wrap_image.squeeze()

    x,y,z = input_fixed_arr.shape

    fig = plt.figure(figsize=(24, 24))
    ax331 = fig.add_subplot(331)
    # ax331.axis('off')
    plt.title('fixed')
    ax331.imshow(input_fixed_arr[int(x / 2), :, :], cmap='Greys')
    ax332 = fig.add_subplot(332)
    # ax332.axis('off')
    plt.title('moving')
    ax332.imshow(moving_image[int(x / 2), :, :], cmap='Greys')
    ax333 = fig.add_subplot(333)
    # ax333.axis('off')
    plt.title('wrapped')
    ax333.imshow(wrap_image[int(x / 2), :, :], cmap='Greys')
    # axial
    ax334 = fig.add_subplot(334)
    # ax334.axis('off')
    ax334.imshow(input_fixed_arr[:, int(y / 2), :], cmap='Greys')
    ax335 = fig.add_subplot(335)
    # ax335.axis('off')
    ax335.imshow(moving_image[:, int(y / 2), :], cmap='Greys')
    ax336 = fig.add_subplot(336)
    # ax336.axis('off')
    ax336.imshow(wrap_image[:, int(y / 2), :], cmap='Greys')
    # coronal
    ax337 = fig.add_subplot(337)
    # ax337.axis('off')
    ax337.imshow(input_fixed_arr[:, :, int(z / 2)], cmap='Greys')
    ax338 = fig.add_subplot(338)
    # ax338.axis('off')
    ax338.imshow(moving_image[:, :, int(z / 2)], cmap='Greys')
    ax339 = fig.add_subplot(339)
    # ax339.axis('off')
    ax339.imshow(wrap_image[:, :, int(z / 2)], cmap='Greys')
    # save fig
    dir_path = os.path.abspath(dir_path)
    result_dir = os.path.join(dir_path, 'evaluation_visual_result')
    if not os.path.exists(result_dir):
        Path(result_dir).mkdir()
    result_case_dir = os.path.join(result_dir, str(case_idx))
    if not os.path.exists(result_case_dir):
        Path(result_case_dir).mkdir()
    fig.savefig(os.path.join(result_case_dir, 'Epoch_{}.jpg'.format(current_epoch)))

    plt.close(fig)

def brain_preProcess(atlas_vol,reduce_size):

    #atlas_vol = atlas_vol[48:-48, 31:-33, 3:-29]
    #atlas_vol = (atlas_vol - np.min(atlas_vol)) / (np.max(atlas_vol) - np.min(atlas_vol))
    atlas_vol = atlas_vol[np.newaxis, np.newaxis, ...]

    atlas_vol = torch.from_numpy(atlas_vol).float()

    B, C, x, y, z = atlas_vol.shape

    if reduce_size:
        atlas_vol = nnf.interpolate(atlas_vol, size=(int(x / 2), int(y / 2), int(z / 2)))

    return atlas_vol

def seg_preProcess(atlas_vol,reduce_size):
    #atlas_vol = atlas_vol[48:-48, 31:-33, 3:-29]
    atlas_vol = atlas_vol[np.newaxis, np.newaxis, ...]

    atlas_vol = torch.from_numpy(atlas_vol).float()
    B, C, x, y, z = atlas_vol.shape

    if reduce_size:
        atlas_vol = nnf.interpolate(atlas_vol, size=(int(x / 2), int(y / 2), int(z / 2)), mode='nearest')
    return atlas_vol

def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''

    vol1 = vol1.squeeze()
    vol2 = vol2.squeeze()
    vol1 = vol1.astype(np.int16)
    vol2 = vol2.astype(np.int16)
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)

def get_neg_jac_fraction(device,flow):
    ps=Poisson(device=device)
    B,C,D,H,W=flow.shape
    phi_X_dx_f, phi_X_dy_f, phi_X_dz_f, phi_Y_dx_f, phi_Y_dy_f, phi_Y_dz_f, phi_Z_dx_f, phi_Z_dy_f, phi_Z_dz_f = ps.Grid2Jac(
        flow, B, C, D, H, W)
    Jac_final = ps.Jac_reshape_img2mat(phi_X_dx_f, phi_X_dy_f, phi_X_dz_f, phi_Y_dx_f, phi_Y_dy_f, phi_Y_dz_f, phi_Z_dx_f,
                                    phi_Z_dy_f, phi_Z_dz_f, B, C, D, H, W)
    Jac_det = torch.det(Jac_final)
    neg_jac = Jac_det<0
    return torch.sum(neg_jac)/(D*H*W)

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)



    
    