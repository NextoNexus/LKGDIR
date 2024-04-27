from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
import nibabel as nib
import numpy as np
import pickle
from reusable import brain_preProcess, seg_preProcess

class OasisDataset_for_img2atlas(Dataset):
    def __init__(self, slurm_data_path, cases_dir, need_seg, transform=None):
        self.slurm_data_path=slurm_data_path
        self.cases_dir=cases_dir
        self.need_seg=need_seg
        self.transform = transform

        self.cases_path=os.path.join(self.slurm_data_path,self.cases_dir)
        self.cases=os.listdir(self.cases_path)
        self.vol_files=[os.path.join(self.cases_path, i, 'brain_affine.nii.gz') for i in self.cases]
        self.seg_files=[os.path.join(self.cases_path, i, 'aseg_affine.nii.gz') for i in self.cases]


    def __len__(self):
        return len(self.vol_files)

    def __getitem__(self, idx):
        print('Loading a img, from case {}...'.format(self.cases[idx]))
        img=nib.load(self.vol_files[idx]).get_fdata()
        img = brain_preProcess(img).squeeze(0)
        if self.transform:
            img = self.transform(img)

        if self.need_seg:
            print('loading a segmentation, from case {}...'.format(self.cases[idx]))
            seg = nib.load(self.seg_files[idx]).get_fdata()
            #print('Seg loaded, grays range from {} to {}'.format(np.min(seg), np.max(seg)))
            #print('Seg preprocess...')
            seg = seg_preProcess(seg)
            seg = seg[np.newaxis, ...]
            return img, seg
        return img

class IxIDataset_for_atlas2img(Dataset):
    def __init__(self, slurm_data_path, cases_dir, need_seg, reduce_size):
        self.slurm_data_path=slurm_data_path
        self.cases_dir=cases_dir
        self.need_seg=need_seg
        self.reduce_size=reduce_size

        self.vol_cases_path = glob.glob(os.path.join(self.slurm_data_path, self.cases_dir, 'vol_subject_*.nii.gz'))
        self.seg_cases_path = glob.glob(os.path.join(self.slurm_data_path, self.cases_dir, 'seg_subject_*.nii.gz'))

    def __len__(self):
        return len(self.vol_cases_path)

    def __getitem__(self, idx):
        print('Loading a img, from case', os.path.split(self.vol_cases_path[idx])[1],'...')

        img = nib.load(self.vol_cases_path[idx]).get_fdata()
        seg = nib.load(self.seg_cases_path[idx]).get_fdata()
        img = brain_preProcess(img, self.reduce_size).squeeze(0)
        seg = seg_preProcess(seg,self.reduce_size).squeeze(0)

        if self.need_seg:
            return img, seg
        return img

