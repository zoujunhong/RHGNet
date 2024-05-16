from .StyleGANGenerator import Decoder
from .networks import TransformerDecoderLayer, HardSoftmax
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from .resnet import ResNet
from sklearn.cluster import AgglomerativeClustering
import cv2
class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 planes,
                 upsample=False):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.norm1 = nn.GroupNorm(1, planes)
        self.norm2 = nn.GroupNorm(1, planes)

        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)

        self.relu = nn.GELU()
        self.upsample = nn.Sequential(
            nn.Conv2d(planes, planes//2, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)) if upsample else nn.Identity()

    def forward(self, x): # slot shape [b*n,c], x shape [b*n,c,h,w]
        """Forward function."""

        def _inner_forward(x):
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            return self.upsample(out + x)

        out = _inner_forward(x)
        out = self.relu(out)
        return out
    
def load_and_freeze(model: nn.Module, dict_name):
    dict = torch.load(dict_name, map_location='cpu')
    model.load_state_dict(dict, strict=True)
    stop_grad(model)

def stop_grad(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m


class SlotAttention(nn.Module):
    def __init__(
        self,
        slot_size, 
        mlp_size, 
        feat_size,
        num_slots=11,
        epsilon=1e-6,
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = epsilon
        self.num_iters = 3

        self.norm_feature = nn.LayerNorm(feat_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_k = linear(feat_size, slot_size, bias=False)
        self.project_v = linear(feat_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.slots_init = nn.Embedding(num_slots, slot_size)
        nn.init.xavier_uniform_(self.slots_init.weight)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_size, slot_size))

    def forward(self, features, sigma):
        B = features.shape[0]
        mu = self.slots_init.weight.expand(B, -1, -1)
        z = torch.randn_like(mu).type_as(features)
        slots_init = mu + z * sigma * mu.detach()
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        # Multiple rounds of attention.
        for i in range(self.num_iters):
            if i == self.num_iters - 1:
                slots = slots.detach() + slots_init - slots_init.detach()
                
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits= torch.einsum('bid,bjd->bij', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn_wm = attn / attn_sum 
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)            

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = nn.Parameter(build_grid(resolution),requires_grad=True)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution=128, num_slots=11, num_iterations=3, hid_dim=64):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.hw = resolution//4
        self.slot_dim = hid_dim

        self.encoder = ResNet(depth=34)
        self.encoder_pos = SoftPositionEmbed(hid_dim, [self.hw,self.hw])
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, 4 * hid_dim),
            nn.LayerNorm(4 * hid_dim),
            nn.GELU(),
            nn.Linear(4 * hid_dim, hid_dim),
            nn.LayerNorm(hid_dim))
        
        self.trans_mlp = nn.Sequential(
            nn.Linear(hid_dim, self.slot_dim),
            nn.LayerNorm(self.slot_dim),
            nn.GELU(),
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.LayerNorm(self.slot_dim, elementwise_affine=False))

        self.decoder = nn.Sequential(
            BasicBlock(64, False),
            BasicBlock(64, True),
            BasicBlock(32, False),
            BasicBlock(32, True),
            BasicBlock(16, False),
            nn.Conv2d(16, 3, 1, 1, 0),
            nn.Sigmoid())
        
        self.slot_attention = SlotAttention(self.slot_dim, self.slot_dim*4, feat_size=self.slot_dim)

        self.decoder_obj = Decoder(slot_dim=64, hid_dim=256, out_dim=3)
        
        self.norm = nn.LayerNorm(64, elementwise_affine=False)


    def forward(self, image, sigma=0):
        # `image` has shape: [batch_size, num_channels, width, height].
        x = self.encoder(image)  # CNN Backbone.
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).contiguous()
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = self.mlp(x)
        x_pix = x.reshape(b,h,w,c).permute(0,3,1,2).contiguous()
        rec_pix = self.decoder(x_pix) * 2 - 1
        
        slots,_ = self.slot_attention(x, sigma=sigma)
        
        slots = self.norm(slots)
        
        slots_bc = slots.reshape((-1, slots.shape[-1]))
        temp = self.decoder_obj(slots_bc)
        recons, masks = temp.reshape(b, -1, temp.shape[1], temp.shape[2], temp.shape[3]).split([3,1], dim=2)
        # `recons` has shape: [batch_size, num_slots, 3, width, height].
        # `masks` has shape: [batch_size, num_slots, 1, width, height].
        recons = torch.sigmoid(recons) * 2 - 1
        masks = F.softmax(masks, dim=1)
        rec_obj = torch.sum(recons * masks, dim=1)  # Recombine image.

        masks_label = masks.detach()
        top_feat = (slots[:,:,:,None,None] * masks_label).sum(dim=1) # [B,N,D]
        top_feat = F.interpolate(top_feat, (self.hw, self.hw), mode='area').detach().flatten(2,3).permute(0,2,1)
        # [batch_size, num_slots, width, height]

        feat2 = self.encoder(rec_obj.detach())  # CNN Backbone.
        feat2 = feat2.permute(0,2,3,1).contiguous()
        feat2 = self.encoder_pos(feat2)
        feat2 = torch.flatten(feat2, 1, 2)
        feat2 = self.mlp(feat2) # [B,N,D]
        x_2 = feat2.reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()
        rec_2 = self.decoder(x_2) * 2 - 1
        
        F_I = self.trans_mlp(feat2)
        loss_td = torch.mean(1 - torch.cosine_similarity(F_I, top_feat, dim=-1))
        return rec_obj, masks.squeeze(2), F.l1_loss(rec_2, rec_obj.detach()) + F.l1_loss(rec_pix, image), loss_td # , loss_feat
    
    def forward_test(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].
        x = self.encoder(image)  # CNN Backbone.
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).contiguous()
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = self.mlp(x)
        x_pix = x.reshape(b,h,w,c).permute(0,3,1,2).contiguous()
        rec_pix = self.decoder(x_pix) * 2 - 1
        
        slots,_ = self.slot_attention(x, sigma=0.0)
        slots = self.norm(slots)
        # slots = slots[:,0:1,:]
        slots_sim = torch.abs(torch.matmul(slots, slots.permute(0,2,1)) / self.slot_dim)
        
        slots_bc = slots.reshape((-1, slots.shape[-1]))
        temp = self.decoder_obj(slots_bc)
        recons, masks = temp.reshape(b, -1, temp.shape[1], temp.shape[2], temp.shape[3]).split([3,1], dim=2)
        # `recons` has shape: [batch_size, num_slots, 3, width, height].
        # `masks` has shape: [batch_size, num_slots, 1, width, height].
        recons = torch.sigmoid(recons)
        masks = F.softmax(masks, dim=1)
        rec_obj = torch.sum(recons * masks, dim=1)  # Recombine image.
        
        masks_label = F.interpolate(masks.squeeze(2), size=(self.hw, self.hw), mode='area')
        top_feat = (self.norm(slots)[:,:,:,None,None] * masks_label.unsqueeze(2)).sum(dim=1).detach() # [B,N,D]
        # [batch_size, num_slots, width, height]

        feat2 = self.encoder(rec_obj.detach())  # CNN Backbone.
        feat2 = feat2.permute(0,2,3,1).contiguous()
        feat2 = self.encoder_pos(feat2)
        feat2 = torch.flatten(feat2, 1, 2)
        feat2 = self.mlp(feat2) # [B,N,D]
        x_2 = feat2.reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()
        rec_2 = self.decoder(x_2) * 2 - 1
        
        x_3 = self.trans_mlp(feat2).reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()
        loss_td = torch.mean(1 - torch.cosine_similarity(x_3, top_feat, dim=1))
        
        x_4 = self.trans_mlp(x).reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()

        return torch.cat([self.trans_mlp(x), self.trans_mlp(feat2)], dim=1), slots, rec_obj, masks, 1 - torch.cosine_similarity(x_3, x_4, dim=1)
    
    def forward_iter(self, image, max_iter=10, sigma=0.):
        # `image` has shape: [batch_size, num_channels, width, height].
        x = self.encoder(image)  # CNN Backbone.
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).contiguous()
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = self.mlp(x)

        F_I = self.trans_mlp(x) # [B, N, D]
        F_dist = 1 - torch.cosine_similarity(F_I.unsqueeze(1), F_I.unsqueeze(2), dim=-1) # [1,N,N]
        th = 0.5
        
        slots,_ = self.slot_attention(x, sigma=sigma)
        slots = self.norm(slots)
        
        for i in range(max_iter):
            dist = 1 - torch.cosine_similarity(F_I.unsqueeze(1), slots.unsqueeze(2), dim=-1) # [1,K,N]
            conflict = torch.min(dist, dim=1)[0] # [1,N]
            # labels = np.array(conflict.cpu()).reshape(32,32)
            # labels = cv2.resize(labels, (128,128), interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite('demo/labels.png', labels * 255)

            # a = input("input:")
            # if a == 0:
            #     break
            
            if torch.max(conflict)<th:
                break
            
            idx = torch.argmax(conflict, dim=1)
            set = []
            for i in range(F_dist.shape[2]):
                if F_dist[0,idx,i] < 0.5 * th:
                    set.append(F_I[:, i:i+1])
            set = torch.cat(set, dim=1).mean(dim=1, keepdim=True)
            slots = torch.cat([slots, set], dim=1)
            
        
        slots_bc = slots.reshape((-1, slots.shape[-1]))
        temp = self.decoder_obj(slots_bc)
        recons, masks = temp.reshape(b, -1, temp.shape[1], temp.shape[2], temp.shape[3]).split([3,1], dim=2)
        recons = torch.sigmoid(recons)
        masks = F.softmax(masks, dim=1)
        rec_obj = torch.sum(recons * masks, dim=1)  # Recombine image.
        
        feat2 = self.encoder(rec_obj.detach())  # CNN Backbone.
        feat2 = feat2.permute(0,2,3,1).contiguous()
        feat2 = self.encoder_pos(feat2)
        feat2 = torch.flatten(feat2, 1, 2)
        feat2 = self.mlp(feat2) # [B,N,D]
        F_H = self.trans_mlp(feat2)
        return torch.cat([F_I, F_H], dim=1), slots, rec_obj, recons, masks, 1 - torch.cosine_similarity(F_I, F_H, dim=-1)
