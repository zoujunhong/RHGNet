import cv2
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path as osp

from torchvision import transforms as pth_transforms

class CLEVRTEX(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 data_root='/home/zoujunhong/dataset/CLEVRTex/clevrtex_full/'):
        self.img_root = data_root
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.split = split
        
        self.path = range(10000, 50000) if self.split == 'train' else range(0,10000)
        print('{} images for {}'.format(len(self.path), self.split))

    def __len__(self):
        """Total number of samples of data."""
        return len(self.path)

    def __getitem__(self, i):
        idx = self.path[i]
        filename = osp.join(self.img_root,str(idx//1000),'CLEVRTEX_full_{}.jpg'.format(str(idx).rjust(6, '0')))
        
        img = cv2.imread(filename)[24:216,64:256,:]
        h = 128
        w = 128
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
        img = self.transform(img)
        return img

    
class CLEVRTEX_segmask(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 data_root='/home/zoujunhong/dataset/CLEVRTex/clevrtex_full/'):
        self.img_root = data_root
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.split = split
        
        self.path = range(10000, 50000) if self.split == 'train' else range(0,10000)
        print('{} images for {}'.format(len(self.path), self.split))

    def __len__(self):
        """Total number of samples of data."""
        return len(self.path)

    def __getitem__(self, i):
        idx = self.path[i]
        filename = annoname = osp.join(self.img_root,str(idx//1000),'CLEVRTEX_full_{}.jpg'.format(str(idx).rjust(6, '0')))
        annoname = annoname = osp.join(self.img_root,str(idx//1000),'CLEVRTEX_full_{}_flat.png'.format(str(idx).rjust(6, '0')))
        
        img = cv2.imread(filename)[24:216,64:256,:]
        mask = cv2.imread(annoname)[24:216,64:256,:].astype(np.float32)
        mask = mask[:,:,0] + 256 * mask[:,:,1] + 65536 * mask[:,:,2]
        res = set(mask.flatten().tolist())
        count = 1
        for i in res:
            if i != 0:
                mask[mask == i] = count
                count += 1
        mask = mask.astype(np.uint8)
        h = 128
        w = 128
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)
        return img, mask

class CLEVRTEX_camo(torch.utils.data.Dataset):

    def __init__(self,
                 split='train',
                 data_root='/home/zoujunhong/dataset/CLEVRTex/clevrtex_camo/'):
        self.img_root = data_root
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.split = split
        
        self.path = range(20000)
        print('{} images for {}'.format(len(self.path), self.split))

    def __len__(self):
        """Total number of samples of data."""
        return len(self.path)

    def __getitem__(self, i):
        idx = self.path[i]
        filename = annoname = osp.join(self.img_root,str(idx//1000),'CLEVRTEX_camo_{}.jpg'.format(str(idx).rjust(6, '0')))
        annoname = annoname = osp.join(self.img_root,str(idx//1000),'CLEVRTEX_camo_{}_flat.png'.format(str(idx).rjust(6, '0')))
        
        img = cv2.imread(filename)[24:216,64:256,:]
        mask = cv2.imread(annoname)[24:216,64:256,:].astype(np.float32)
        mask = mask[:,:,0] + 256 * mask[:,:,1] + 65536 * mask[:,:,2]
        res = set(mask.flatten().tolist())
        count = 1
        for i in res:
            if i != 0:
                mask[mask == i] = count
                count += 1
        mask = mask.astype(np.uint8)
        h = 128
        w = 128
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)
        return img, mask

class CLEVRTEX_outd(torch.utils.data.Dataset):

    def __init__(self,
                 split='train',
                 data_root='/home/zoujunhong/dataset/CLEVRTex/clevrtex_outd/'):
        self.img_root = data_root
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.split = split
        
        self.path = range(10000)
        print('{} images for {}'.format(len(self.path), self.split))

    def __len__(self):
        """Total number of samples of data."""
        return len(self.path)

    def __getitem__(self, i):
        idx = self.path[i]
        filename = annoname = osp.join(self.img_root,str(idx//1000),'CLEVRTEX_outd_{}.jpg'.format(str(idx).rjust(6, '0')))
        annoname = annoname = osp.join(self.img_root,str(idx//1000),'CLEVRTEX_outd_{}_flat.png'.format(str(idx).rjust(6, '0')))
        
        img = cv2.imread(filename)[29:221,64:256,:]
        mask = cv2.imread(annoname)[29:221,64:256,:].astype(np.float32)
        mask = mask[:,:,0] + 256 * mask[:,:,1] + 65536 * mask[:,:,2]
        res = set(mask.flatten().tolist())
        count = 1
        for i in res:
            if i != 0:
                mask[mask == i] = count
                count += 1
        mask = mask.astype(np.uint8)
        h = 128
        w = 128
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)
        return img, mask


