import numpy as np
from torch import nn
import torch
from .networks import TransformerEncoderLayer, TransformerDecoderLayer

class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 slot_dim,
                 planes,
                 upsample=False):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.style = nn.Sequential(
            nn.Linear(slot_dim, 2*planes),
            nn.GELU(),
            nn.Linear(2*planes, 2*planes))
        self.norm = nn.GroupNorm(1, planes)
        self.norm1 = nn.GroupNorm(1, planes)
        self.norm2 = nn.GroupNorm(1, planes)

        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)

        self.relu = nn.GELU()
        self.upsample = nn.Sequential(
            nn.Conv2d(planes, planes//2, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ) if upsample else nn.Identity()

    def forward(self, x, slot): # slot shape [b*n,c], x shape [b*n,c,h,w]
        """Forward function."""

        def _inner_forward(x):
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            return self.upsample(out + x)
        
        gain, bias = torch.split(self.style(slot), [self.planes,self.planes], dim = -1)
        x = (1 + gain[:,:,None,None]) * self.norm(x) + bias[:,:,None,None]
        out = _inner_forward(x)
        out = self.relu(out)
        return out

class BasicBlock_tf(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 slot_dim,
                 planes):
        super(BasicBlock_tf, self).__init__()
        self.planes = planes
        self.style = nn.Linear(slot_dim, 2*planes)
        self.norm = nn.LayerNorm(planes)
        self.attn = TransformerEncoderLayer(planes, planes//32, 0)
        self.ffn = nn.Sequential(
            nn.LayerNorm(planes),
            nn.Linear(planes, 4*planes),
            nn.GELU(),
            nn.Linear(4*planes, planes))

    def forward(self, x, slot): # slot shape [B*N,C], x shape [B*N,L,C]
        """Forward function."""
        gain, bias = torch.split(self.style(slot), [self.planes,self.planes], dim = -1)
        x = (1 + gain[:,None,:]) * self.norm(x) + bias[:,None,:]
        out = self.attn(x)
        out = self.ffn(out) + out
        return out

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).permute(0,3,1,2).contiguous()

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.embedding = nn.Conv2d(4, hidden_size, 1, 1, 0, bias=True)
        self.grid = nn.Parameter(build_grid(resolution),requires_grad=True)

    def forward(self):
        grid = self.embedding(self.grid)
        return grid


class Decoder(nn.Module):
    def __init__(self, slot_dim=512, hid_dim=64, out_dim=3, resolution=8, block_num=4, target_resolution=128):
        super().__init__()
        assert 2**block_num * resolution >= target_resolution
        self.grid = nn.Parameter(build_grid([resolution, resolution]), requires_grad=True) # [1,4,h,w]
        self.embedding = nn.Conv2d(4, hid_dim, 1, 1, 0)
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            upsample = resolution < target_resolution
            self.generator_blocks.append(BasicBlock(slot_dim, hid_dim, upsample=upsample))
            hid_dim = hid_dim//2 if upsample else hid_dim
            resolution = resolution*2 if upsample else resolution
        
        self.end_cnn = nn.Sequential(
            nn.Conv2d(hid_dim, max(hid_dim, out_dim+1), 1, 1, 0),
            nn.GroupNorm(1, max(hid_dim, out_dim)),
            nn.GELU(),
            nn.Conv2d(max(hid_dim, out_dim+1), out_dim+1, 1, 1, 0))

    def forward(self, slots):
        b = slots.shape[0]
        init_grid = torch.repeat_interleave(self.embedding(self.grid),b,0)
        for i in range(len(self.generator_blocks)):
            init_grid = self.generator_blocks[i](init_grid, slots)

        return self.end_cnn(init_grid)
    
class Decoder_clevr(nn.Module):
    def __init__(self, slot_dim=512, hid_dim=64, out_dim=3, resolution=8, block_num=4, target_resolution=128):
        super().__init__()
        assert 2**block_num * resolution >= target_resolution
        self.grid = nn.Parameter(build_grid([resolution, resolution]), requires_grad=True) # [1,4,h,w]
        self.embedding = nn.Conv2d(4, hid_dim, 1, 1, 0)
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            upsample = resolution < target_resolution
            self.generator_blocks.append(BasicBlock(slot_dim, hid_dim, upsample=upsample))
            hid_dim = hid_dim//2 if upsample else hid_dim
            resolution = resolution*2 if upsample else resolution
        
        self.end_cnn = nn.Sequential(
            nn.Conv2d(hid_dim, max(hid_dim, out_dim+1), 3, 1, 1),
            nn.GroupNorm(1, max(hid_dim, out_dim)),
            nn.GELU(),
            nn.Conv2d(max(hid_dim, out_dim+1), out_dim+1, 1, 1, 0))

    def forward(self, slots):
        b = slots.shape[0]
        init_grid = torch.repeat_interleave(self.embedding(self.grid),b,0)
        for i in range(len(self.generator_blocks)):
            init_grid = self.generator_blocks[i](init_grid, slots)

        return self.end_cnn(init_grid)

class Decoder_tf(nn.Module):
    def __init__(self, slot_dim=512, hid_dim=384, out_dim=384, resolution=[8,8], block_num=4):
        super().__init__()
        self.grid = nn.Parameter(torch.randn(1,resolution[0]*resolution[1], hid_dim), requires_grad=True) # [1,4,h,w]
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            self.generator_blocks.append(TransformerDecoderLayer(hid_dim, slot_dim, hid_dim//64, attention_dropout=0))
            
        self.end_cnn = nn.Sequential(
            nn.Linear(hid_dim, max(hid_dim, out_dim)),
            nn.LayerNorm(max(hid_dim, out_dim)),
            nn.GELU(),
            nn.Linear(max(hid_dim, out_dim), out_dim))

    def forward(self, slots):
        b = slots.shape[0]
        init_grid = torch.repeat_interleave(self.grid,b,0)
        for i in range(len(self.generator_blocks)):
            attn, init_grid = self.generator_blocks[i](slots, init_grid)

        return self.end_cnn(init_grid), attn
