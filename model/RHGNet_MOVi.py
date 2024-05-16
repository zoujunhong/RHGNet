from .GansformerGenerator import Generator
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from .dino import vit_tiny
from .networks import HardSoftmax
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
    def __init__(self, resolution=224, num_slots=11, num_iterations=3, hid_dim=192):
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
        self.slot_dim = 128
        self.hw = resolution//16

        self.encoder = vit_tiny()
        self.trans_mlp = nn.Sequential(
            nn.Conv2d(192, 512, 1, 1, 0),
            BasicBlock(512, True),
            BasicBlock(256, True),
            BasicBlock(128, False),
            nn.GroupNorm(1, self.slot_dim, affine=False))
        
        self.decoder = nn.Sequential(
            BasicBlock(192, True),
            BasicBlock(96, True),
            BasicBlock(48, True),
            BasicBlock(24, True),
            BasicBlock(12, False),
            nn.Conv2d(12, 3, 1, 1, 0))
        
        self.slot_attn = SlotAttention(self.slot_dim, self.slot_dim*4, feat_size=hid_dim)

        self.generator = Generator(slot_dim=self.slot_dim, base_dim=512, block_num=4)

        self.norm = nn.LayerNorm(self.slot_dim, elementwise_affine=False)


    def forward(self, image, sigma=0):
        # encoder
        feat = self.encoder(image)  # CNN Backbone.
        b,n,c = feat.shape
        x_pix = feat.reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()
        rec_pix = self.decoder(x_pix)
        slots,_ = self.slot_attn(feat, sigma=sigma)
        slots = self.norm(slots)
        slots_sim = torch.abs(torch.matmul(slots, slots.permute(0,2,1)) / self.slot_dim)
        target = torch.eye(self.num_slots).unsqueeze(0).to(slots_sim.device)
        
        rec_obj, attn = self.generator(slots)
        attn = HardSoftmax(attn, dim=-1)
        top_feat = (slots.permute(0,2,1).unsqueeze(2) * attn).sum(dim=-1).permute(0,2,1).detach() # [B,N,D]

        feat2 = self.encoder(rec_obj.detach()).reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()  # CNN Backbone.
        rec_2 = self.decoder(feat2)
        
        F_I = self.trans_mlp(feat2).flatten(2,3).permute(0,2,1)
        loss_td = torch.mean(1 - torch.cosine_similarity(F_I, top_feat, dim=-1)) + F.mse_loss(slots_sim, target)
        return rec_obj, F.l1_loss(rec_2, rec_obj.detach()) + F.l1_loss(rec_pix, image), loss_td # , loss_feat
    
    def forward_test(self, image):
        # encoder
        feat = self.encoder(image)  # CNN Backbone.
        b,n,c = feat.shape
        x_pix = feat.reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()
        slots,_ = self.slot_attn(feat, sigma=0)
        slots = self.norm(slots)
        
        rec_obj, attn = self.generator(slots)

        feat2 = self.encoder(rec_obj.detach())  # CNN Backbone.
        
        x_3 = self.trans_mlp(feat2.reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous())
        x_4 = self.trans_mlp(x_pix)
        return torch.cat([x_4, x_3], dim=-1).flatten(2,3).permute(0,2,1), slots, rec_obj, attn, 1 - torch.cosine_similarity(x_3, x_4, dim=1)
    
    def forward_iter(self, image):
        # encoder
        feat = self.encoder(image)  # CNN Backbone.
        b,n,c = feat.shape
        x_pix = feat.reshape(b,self.hw,self.hw,c).permute(0,3,1,2).contiguous()
        F_I = self.trans_mlp(x_pix)
        F_I = F.interpolate(F_I, size=(14,14), mode='area')
        F_I = F_I.flatten(2,3).permute(0,2,1) # [B, N, D]
        F_I_norm = F.normalize(F_I, dim=-1)
        F_dist = 1 - torch.matmul(F_I_norm,F_I_norm.permute(0,2,1)) # [1,N,N]
        
        th = 0.7
        
        slots,_ = self.slot_attn(feat, sigma=0.)
        slots = self.norm(slots)
        slots = slots[:,8:9,:]
        
        for i in range(10):
            slots_norm = F.normalize(slots, dim=-1)
            # dist = 1 - torch.cosine_similarity(F_I.unsqueeze(1), slots.unsqueeze(2), dim=-1) # [1,K,N]
            # print(0)
            dist1 = 1 - torch.matmul(slots_norm, F_I_norm.permute(0,2,1)) # [1,K,N]
            # print(1)
            # print(dist.shape, dist1.shape)
            # print(torch.mean((dist-dist1)**2))
            conflict = torch.min(dist1, dim=1)[0] # [1,N]
            # labels = np.array(conflict.cpu()).reshape(14,14)
            # labels = cv2.resize(labels, (224,224), interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite('demo/labels.png', labels * 255)
            # print(2)
            # a = input("input:")
            # if a == 0:
            #     break
            
            if torch.max(conflict)<th:
                break
            
            idx = torch.argmax(conflict, dim=1)
            set = None
            count = 0
            # print(F_dist.shape)
            for i in range(F_dist.shape[2]):
                if F_dist[0,idx,i] < th:
                    set = F_I[:, i:i+1] if set is None else set + F_I[:, i:i+1]
                    count += 1
            set = self.norm(F_I[:, idx:idx+1])
            # set = self.norm(torch.cat(set, dim=1).mean(dim=1, keepdim=True))
            # print(idx//32, idx%32)
            slots = torch.cat([slots, set], dim=1)
            
        rec_obj, masks = self.generator(slots)
        
        
        return slots, rec_obj, masks
    
if __name__ == '__main__':
    model = nn.Conv2d(128,128,1,1,0, bias=False) # SlotAttentionAutoEncoder()
    x = torch.randn([1,128,224,224])

    from thop import profile
    Flops, Params = profile(model,(x,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))