import cv2
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path as osp

from torchvision import transforms as pth_transforms

class CLEVRDataset(torch.utils.data.Dataset):

    def __init__(self,
                 split='train',
                 data_root='/home/zoujunhong/dataset/CLEVR_with_mask/'):
        self.img_root = data_root
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.split = split
        
        self.path = range(70000) if self.split == 'train' else range(70000,100000) # np.load('CLEVR6_val_idx.npy').tolist()
        print('{} images for {}'.format(len(self.path), self.split))


    def __len__(self):
        """Total number of samples of data."""
        return len(self.path)

    def __getitem__(self, i):
        idx = self.path[i]
        filename = osp.join(self.img_root,'image','clevr_{}.jpg'.format(idx))
        
        img = cv2.imread(filename)[24:216,64:256,:]
        h = 128
        w = 128
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
        img = self.transform(img)
        return img
    

class CLEVRDataset_segmask(torch.utils.data.Dataset):

    def __init__(self,
                 split='train',
                 data_root='/home/zoujunhong/dataset/CLEVR_with_mask/'):
        self.img_root = data_root
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.split = split
        
        self.path = range(70000) if self.split == 'train' else range(70000, 100000) # np.load('CLEVR6_val_idx.npy').tolist()
        print('{} images for {}'.format(len(self.path), self.split))


    def __len__(self):
        """Total number of samples of data."""
        return len(self.path)

    def __getitem__(self, i):
        idx = self.path[i]
        filename =  osp.join(self.img_root,'image','clevr_{}.jpg'.format(idx))
        annoname = osp.join(self.img_root,'object_mask','clevr_{}.png'.format(idx))
        
        img = cv2.imread(filename)[24:216,64:256,:]
        mask = cv2.imread(annoname)[24:216,64:256,0]
        h = 128
        w = 128
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)
        return img, mask
