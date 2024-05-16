# System libs
import os
import time
import json
# import math
import random
import argparse

# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
# Our libs
from dataset.CLEVRTex import CLEVRTEX_segmask as CLEVRDataset
# from dataset_1130 import CLEVRDataset_segmask as CLEVRDataset
from model.RHGNet import SlotAttentionAutoEncoder as Model
# from model.slot_attention_ablationtd import SlotAttentionAutoEncoder as RH_Capsule
import numpy as np
import datetime as datetime
from metrics.OCL import ObjectIOU
from metrics.OCL import ARI
from metrics.OCL import MSC as mBO

seed_value = 42# 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
# train one epoch
from random import randint
palette = [np.array([randint(1,255),randint(1,255),randint(1,255)]) for i in range(255)]
def test(segmentation_module: Model, data_loader):

    segmentation_module.eval()
    total_ARI = 0.
    total_IOU = 0.
    total_mse = 0.
    total_img = 0.
    torch.set_printoptions(precision=2,sci_mode=False,linewidth=100)
    for i,data in enumerate(data_loader):
        with torch.no_grad():
            imgs, masks = data
            origin_img = imgs.clone()
            origin_img = ((origin_img+1)*127.5).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
                
            imgs = imgs.cuda()

            # forward pass
            feat, slots, rec_obj, subpart_rec, subpart_mask, conflict = segmentation_module.forward_iter(imgs, max_iter=2)
                
            mse = F.mse_loss(rec_obj, (imgs+1)/2) * 128**2 # computing mse after normalized to [0,1]
            
            # ignore=0 when excluding background and ignore=100 when including
            object_mask = subpart_mask.argmax(dim=1).squeeze()
            ari = ARI(object_mask, masks.squeeze().cuda(), ignore=0)
            iou,_,_,_ = ObjectIOU(object_mask, masks.squeeze().cuda(), ignore=100)
            if not torch.isnan(ari):
                total_ARI += ari
                total_IOU += iou
                total_mse += mse
                total_img += 1
            print('avg_ARI:{:.4f}, avg_iou:{:.4f}, avg_mse:{:.2f}'.format(total_ARI/total_img,total_IOU/total_img,total_mse/total_img))

            '''
            un-comment the following code for visualization
            '''
            # object_mask = object_mask.cpu().numpy().astype(np.uint8) * 20
            # object_mask = np.expand_dims(object_mask, axis=-1).repeat(3, axis=-1)
            # object_mask_palette = object_mask.copy()
            # for i in range(object_mask.shape[0]):
            #     for j in range(object_mask.shape[1]):
            #          object_mask_palette[i,j,:] = palette[object_mask[i,j,0]]
                
            # masks = masks.squeeze().numpy().astype(np.uint8) * 20
            # masks = np.expand_dims(masks, axis=-1).repeat(3, axis=-1)
            # masks_palette = masks.copy()
            # for i in range(masks.shape[0]):
            #     for j in range(masks.shape[1]):
            #          masks_palette[i,j,:] = palette[masks[i,j,0]]
                
            # rec = ((rec_obj.clamp(0,1).squeeze().permute(1,2,0).cpu().numpy())*255).astype(np.uint8)
            # # rec = ((rec_obj.clamp(-1,1).squeeze().permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            # to_save = np.concatenate((origin_img, rec, object_mask_palette*0.5 + rec * 0.5, masks_palette, object_mask_palette), axis=1)
            # cv2.imwrite('demo/mask.png', to_save)
                
            # for i in range(subpart_mask.shape[1]):
            #     mask = subpart_mask[:,i,...].squeeze().unsqueeze(-1).repeat(1,1,3).detach().cpu().numpy()
            #     rec = (subpart_rec[:,i,...] + 1)*127.5
            #     rec = rec.squeeze().permute(1,2,0).detach().cpu().numpy()
            #     # mask[mask < 0.95] = 0
            #     # print(i,torch.max(subpart_mask[:,i,...]))
            #     cv2.imwrite('demo/mask_{}.png'.format(i), (mask*rec + (1-mask)*128).astype(np.uint8))

            # a = input("input:")
            # if a == 0:
            #     break

def main():
    # Network Builders
    model = Model(num_slots=11)
    print('pretrained model loaded !')
    dataset_train = CLEVRDataset(split='val')
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, pin_memory=True,
                                    num_workers=0)
    
    # load nets into gpu
    to_load = torch.load('checkpoint/clevrtex.pth',map_location=torch.device("cpu"))
    model.load_state_dict(to_load,strict=False)
    model = model.cuda()
    test(model, loader_train)


    print('Testing Done!')


if __name__ == '__main__':
    main()
