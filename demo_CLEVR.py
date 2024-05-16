# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# Our libs
from model.RHGNet import SlotAttentionAutoEncoder as Model
from dataset.CLEVR import CLEVRDataset_segmask as CLEVRDataset

from metrics.OCL import ObjectIOU as mIoU
from metrics.OCL import ARI
from metrics.OCL import MSC as mBO
from random import randint

palette = [np.array([randint(1,255),randint(1,255),randint(1,255)]) for i in range(255)]
# train one epoch
def test(segmentation_module: Model, data_loader):

    segmentation_module.eval()
    total_mse = 0.
    total_ARI = 0.
    total_IOU = 0.
    total_img = 0.
    total_img2 = 0.
    torch.set_printoptions(precision=2,sci_mode=False,linewidth=1000)
    for i,data in enumerate(data_loader):

        with torch.no_grad():
            imgs, masks = data
            masks = masks.squeeze().cuda()
            origin_img = ((imgs+1)*127.5).squeeze().permute(1,2,0).cpu().numpy()
                    
            imgs = imgs.cuda()

            # forward pass
            # _,_,rec,subpart_mask,_ = segmentation_module.forward_test(imgs)
            feat, slots, rec, subpart_rec, subpart_mask, conflict = segmentation_module.forward_iter(imgs, max_iter=2)
                
            object_mask = subpart_mask.squeeze().argmax(dim=0)
            mse = F.mse_loss((imgs+1)/2, rec) * 128**2
            ari = ARI(object_mask, masks, ignore=0)
            if not torch.isnan(ari):
                total_ARI += ari
                total_mse += mse
                total_img += 1
            print('{},avg_ARI:{}'.format(i,total_ARI/total_img))
            print('{},avg_mse:{}'.format(i+1,total_mse/total_img))

            OIOU, c1, IOU_list, Pix_list = mIoU(object_mask, masks, ignore=100)
            oiou = OIOU/c1
            if not torch.isnan(oiou):
                total_IOU += oiou
                total_img2 += 1
                print('{},{}'.format(i, total_IOU/total_img2))
    
            '''
            un-comment the following code for visualization
            '''
            # object_mask = object_mask.cpu().numpy().astype(np.uint8) * 20
            # object_mask = np.expand_dims(object_mask, axis=-1).repeat(3, axis=-1)
            # object_mask_palette = object_mask.copy()
            # for i in range(object_mask.shape[0]):
            #     for j in range(object_mask.shape[1]):
            #         object_mask_palette[i,j,:] = palette[object_mask[i,j,0]]

            # masks = masks.cpu().squeeze().numpy().astype(np.uint8) * 20
            # masks = np.expand_dims(masks, axis=-1).repeat(3, axis=-1)
            # masks_palette = masks.copy()
            # for i in range(masks.shape[0]):
            #     for j in range(masks.shape[1]):
            #         masks_palette[i,j,:] = palette[masks[i,j,0]]

            # object_rec_slot  = (rec.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            # to_save = np.concatenate((origin_img, object_rec_slot, masks, object_mask), axis=1)
            # cv2.imwrite('demo/mask.png', to_save)

            # a = input("input:")
            # if a == 0:
            #     break
        

def main():
    # Network Builders
    segmentation_module = Model(num_slots=11)
    dataset_train = CLEVRDataset(split='val', data_root='./CLEVR_with_mask/')
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, pin_memory=True,
                                    num_workers=0)
    
    # load nets into gpu
    segmentation_module = segmentation_module.cuda()
    to_load = torch.load('checkpoint/clevr.pth',map_location=torch.device("cuda:0"))
    segmentation_module.load_state_dict(to_load,strict=False)
    
    test(segmentation_module, loader_train)

    print('Testing Done!')


if __name__ == '__main__':
    main()
