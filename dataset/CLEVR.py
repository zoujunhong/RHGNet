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
        
        self.path = range(40000) if self.split == 'train' else range(40000,50000) # np.load('CLEVR6_val_idx.npy').tolist()
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

    
class CLEVRTEX_slots(torch.utils.data.Dataset):

    def __init__(self):
        self.slots = np.load('clevrtex_slots.npy')
        self.slots = self.slots.reshape(-1, 11, 64)

    def __len__(self):
        """Total number of samples of data."""
        return self.slots.shape[0]

    def __getitem__(self, i):
        slot = np.copy(self.slots[i])
        return torch.from_numpy(slot)

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
        
        self.path = range(10000, 50000) if self.split == 'train' else range(0,10000) # np.load('CLEVR6_val_idx.npy').tolist()
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


class MOVi(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/home/zoujunhong/dataset/movi_c',
                 split='train'):
        self.img_root = osp.join(data_root,split,'video')
        self.img_infos = sorted(os.listdir(self.img_root))

        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform2 = pth_transforms.Compose([
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
        img = cv2.resize(cv2.imread(filename), (224,224), interpolation=cv2.INTER_AREA)
        
        img1 = self.transform(img)
        img2 = self.transform2(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img2#, torch.from_numpy(ann)


class MOVi_property_predict(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/home/zoujunhong/dataset/movi_c',
                 split='train'):
        self.img_root = osp.join(data_root,split,'video')
        self.ann_root = osp.join(data_root,split,'seg')
        self.inst_root = osp.join(data_root,split,'instance')
        self.img_infos = sorted(os.listdir(self.img_root))

        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        annoname = osp.join(self.ann_root, self.img_infos[vid_idx], str(img_idx)+'.png')
        category = np.load(osp.join(self.inst_root, self.img_infos[vid_idx], 'category.npy'))
        num = category.shape[0]
        category = np.concatenate((category, np.zeros((11-num,))), axis=0)
        bboxes_3d = np.load(osp.join(self.inst_root, self.img_infos[vid_idx], 'bboxes_3d.npy'))[:,img_idx]
        positions = np.load(osp.join(self.inst_root, self.img_infos[vid_idx], 'positions.npy'))[:,img_idx]
        pos = np.concatenate((bboxes_3d.reshape(-1,24), positions), axis=1)
        pos = np.concatenate((pos, np.zeros((11-num,27))), axis=0)
        img = cv2.resize(cv2.imread(filename), (224,224), interpolation=cv2.INTER_AREA)
        ann = cv2.resize(cv2.imread(annoname), (224,224), interpolation=cv2.INTER_NEAREST)[:,:,0]
        
        img1 = self.transform(img)
        return img1, torch.from_numpy(ann), category, pos


class MOVi_SD(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/home/zoujunhong/dataset/movi_c',
                 split='train'):
        self.img_root = osp.join(data_root,split,'video') 
        self.img_infos = sorted(os.listdir(self.img_root))

        self.code_root = 'SD_code_224/'
        self.code_infos = sorted(os.listdir(self.code_root))

        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.code_infos)

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        img = cv2.imread(filename)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        img = self.transform(img)

        latent = np.load(osp.join(self.code_root, str(idx)+'.npy'))
        return img, torch.from_numpy(latent)


class MOVi_Slot(torch.utils.data.Dataset):
    def __init__(self,):
        self.code_root = 'SlotVAE_code/'
        self.code_infos = sorted(os.listdir(self.code_root))

    def __len__(self):
        return len(self.code_infos)

    def __getitem__(self, idx):
        slots = np.load(osp.join(self.code_root, str(idx)+'.npy'))
        return torch.from_numpy(slots)

class MOVi_test(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/home/zoujunhong/dataset/movi_c',
                 split='train'):
        self.img_root = osp.join(data_root,split,'video') 
        self.ann_root = osp.join(data_root,split,'seg') 
        self.img_infos = sorted(os.listdir(self.img_root))

        self.transform2 = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Resize(224),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform1 = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Resize(224),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        # annoname = osp.join(self.ann_root, self.img_infos[vid_idx], str(img_idx)+'.png')
        img = cv2.imread(filename)
        # img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
        # ann = cv2.imread(annoname)[:,:,0]
        # ann = cv2.resize(ann, (224,224), interpolation=cv2.INTER_NEAREST)
        # img1 = self.transform1(img)
        # img2 = self.transform2(img)
        img2 = self.transform2(img)
        return img2 #img1, img2, torch.from_numpy(ann)   

if __name__ == "__main__":
    dataset = MOVi_property_predict()
    for i in range(10):
        dataset.__getitem__(i*24)

