import cv2
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path as osp

from torchvision import transforms as pth_transforms

class MOVi(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/home/zoujunhong/dataset/movi_c',
                 split='train',
                 resolution=224):
        self.img_root = osp.join(data_root,split,'video')
        self.img_infos = sorted(os.listdir(self.img_root))
        self.resolution = resolution
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        img = cv2.resize(cv2.imread(filename), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
        img = self.transform(img)
        return img


class MOVi_test(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/home/zoujunhong/dataset/movi_c',
                 split='train',
                 resolution=224):
        self.img_root = osp.join(data_root,split,'video')
        self.ann_root = osp.join(data_root,split,'seg')
        self.img_infos = sorted(os.listdir(self.img_root))

        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resolution = resolution

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 12

    def __getitem__(self, idx):
        vid_idx = idx // 12
        img_idx = idx % 12 + 6
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        img = cv2.resize(cv2.imread(filename), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        annoname = osp.join(self.ann_root, self.img_infos[vid_idx], str(img_idx)+'.png')
        ann = cv2.imread(annoname)[:,:,0]
        ann = cv2.resize(ann, (self.resolution,self.resolution), interpolation=cv2.INTER_NEAREST)
        return img, torch.from_numpy(ann)