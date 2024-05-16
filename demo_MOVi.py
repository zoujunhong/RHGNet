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
from dataset.MOVi import MOVi_test as Dataset
from model.RHGNet_MOVi import SlotAttentionAutoEncoder as Model
import numpy as np
import datetime as datetime
import cv2
from metrics.OCL import ObjectIOU as metrics
from metrics.OCL import ARI as metrics1
import metrics.OCL as OCL
import lpips
from random import randint
from sklearn.cluster import AgglomerativeClustering
# train one epoch
palette = [np.array([randint(1,255),randint(1,255),randint(1,255)]) for i in range(255)]
def test(segmentation_module: Model, data_loader):
    total_ARI = 0.
    total_IOU = 0.
    total_img = 0.
    total_obj = 0.
    IOU_count = [[],[],[],[],[]]
    segmentation_module.eval()
    num_slots = 11
    res = 56
    init_res = 224
    torch.set_printoptions(precision=4,sci_mode=False,linewidth=1000)
    net = nn.LayerNorm(128, elementwise_affine=False)
    
    # for p in perceptual_loss.parameters():
    #     p.requires_grad = False
    for i,data in enumerate(data_loader):
        # if i < 28:
        #     continue
        with torch.no_grad():
            # while True:
                imgs, masks = data
                origin_img = imgs.clone()
                # origin_img = F.interpolate(origin_img, size=(32,32), mode='bilinear', align_corners=False)
                origin_img = ((origin_img+1)*127.5).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
                
                imgs = imgs.cuda()
                # forward pass
                # feat, slots, rec_obj, subpart_mask, conflict = segmentation_module.forward_test(imgs)
                slots, rec_obj, subpart_mask = segmentation_module.forward_iter(imgs)
                
                mask = subpart_mask.permute(0,3,1,2).reshape(1,subpart_mask.shape[-1],56,56)
                mask = F.interpolate(mask, scale_factor=4, mode='bicubic')
                # print(mask.shape)
                new_mask = mask.squeeze(0).argmax(dim=0).reshape(224,224)
                object_mask = new_mask.clone()

                ari = metrics1(object_mask, masks.squeeze().cuda(), ignore=0)
                # # ari,_,_,_ = metrics(object_mask, masks.squeeze().cuda(), ignore=100)
                if not torch.isnan(ari):
                    total_ARI += ari
                    total_img += 1
                print('{},avg_ARI:{}'.format(i+1,total_ARI/total_img))
                
    #             OIOU, c1, IOU_list, Pix_list = metrics(object_mask, masks.squeeze(), ignore=100)
    #             oiou = OIOU/c1
    #             # if not torch.isnan(OIOU):
    #             total_IOU += OIOU/c1
    #             total_obj += 1
    #             for j in range(len(Pix_list)):
    #                 if Pix_list[j] == 0:
    #                      continue
    #                 idx = Pix_list[j]//100 if Pix_list[j]//100<=4 else 4
                    
    #                 IOU_count[idx].append(IOU_list[j])
    #                 # if Pix_list[j] > 0 and Pix_list[j] <= 150:
    #                 #     IOU_count[0].append(IOU_list[j])
    #                 # elif Pix_list[j] > 150 and Pix_list[j] <= 1000:
    #                 #     IOU_count[1].append(IOU_list[j])
    #                 # elif Pix_list[j] > 1000:
    #                 #     IOU_count[2].append(IOU_list[j])
    #             print('{},{}'.format(i, total_IOU/total_obj))
    
    # # for i in range(5):
    # #     IOU = np.array(IOU_count[i])
    # #     print(np.mean(IOU)*100)
        
    #             # print(i, total_td/(total+1e-8))
                object_mask = object_mask.cpu().numpy().astype(np.uint8) * 20
                object_mask = np.expand_dims(object_mask, axis=-1).repeat(3, axis=-1)
                object_mask_palette = object_mask.copy()
                for i in range(object_mask.shape[0]):
                    for j in range(object_mask.shape[1]):
                        object_mask_palette[i,j,:] = palette[object_mask[i,j,0]]
                
                masks = masks.squeeze().numpy().astype(np.uint8) * 20
                masks = np.expand_dims(masks, axis=-1).repeat(3, axis=-1)
                masks_palette = masks.copy()
                for i in range(masks.shape[0]):
                    for j in range(masks.shape[1]):
                         masks_palette[i,j,:] = palette[masks[i,j,0]]
                
                rec = ((rec_obj.clamp(-1,1).squeeze().permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
                to_save = np.concatenate((origin_img, rec, object_mask_palette*0.5 + origin_img * 0.5, masks_palette*1 + origin_img * 0, object_mask, masks), axis=1)
                cv2.imwrite('demo/mask.png', np.flip(to_save, axis=-1))

                a = input("input:")
                if a == 0:
                    break
    

def main():
    # Network Builders
    model = Model()
    
    dataset_train = Dataset(split='val')
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, pin_memory=False,
                                    num_workers=0)
    
    # load nets into gpu
    to_load = torch.load('checkpoint/movi.pth',map_location=torch.device("cpu"))
    model.load_state_dict(to_load,strict=True)
    model = model.cuda()
    print('pretrained model loaded !')
    test(model, loader_train)


    print('Testing Done!')


if __name__ == '__main__':
    main()
