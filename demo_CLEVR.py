# System libs
import cv2
import argparse

# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
# Our libs
from dataset.CLEVR import CLEVRDataset_segmask as CLEVRDataset
from dataset.CLEVR import CLEVRDemo
from model.RHGNet import SlotAttentionAutoEncoder as Model
import numpy as np
import datetime as datetime
from metrics.OCL import ObjectIOU
from metrics.OCL import ARI
from metrics.OCL import MSC as mBO

# train one epoch
from random import randint
palette = [np.array([randint(1,255),randint(1,255),randint(1,255)]) for i in range(255)]
def test(segmentation_module: Model, data_loader, args):

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
            visualization
            '''
            if args.visualization:
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
                    
                rec = ((rec_obj.clamp(0,1).squeeze().permute(1,2,0).cpu().numpy())*255).astype(np.uint8)
                # rec = ((rec_obj.clamp(-1,1).squeeze().permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
                to_save = np.concatenate((origin_img, rec, object_mask_palette*0.5 + rec * 0.5, masks_palette, object_mask_palette), axis=1)
                cv2.imwrite('demo/mask.png', to_save)

                a = input("input:")
                if a == 0:
                    break

def main():
    # Network Builders
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument("--demo", action='store_true')
    parser.add_argument("--dataroot",type=str,default='./CLEVR_with_mask/')
    parser.add_argument("--checkpoint",type=str,default='checkpoint/clevrtex.pth')
    parser.add_argument("--visualization", action='store_true')
    args = parser.parse_args()

    print(args)
    
    model = Model(num_slots=11)
    dataset_train = CLEVRDataset(split='val', data_root=args.dataroot) if not args.demo else CLEVRDemo(data_root=args.dataroot)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
    
    # load nets into gpu
    to_load = torch.load(args.checkpoint ,map_location=torch.device("cpu"))
    model.load_state_dict(to_load,strict=False)
    model = model.cuda()
    test(model, loader_train, args)

    print('Testing Done!')


if __name__ == '__main__':
    main()
