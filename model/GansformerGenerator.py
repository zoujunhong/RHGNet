import numpy as np
from torch import nn
import torch
from .networks import SimplexAttention, TransformerDecoderLayer

class ResLayer(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 planes,
                 upsample=False):
        super(ResLayer, self).__init__()
        self.planes = planes
        self.norm1 = nn.GroupNorm(1, planes)
        self.norm2 = nn.GroupNorm(1, planes)

        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)

        self.relu = nn.GELU()
        self.upsample = nn.Sequential(
            nn.Conv2d(planes, planes//2, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)) if upsample else nn.Identity()
        

    def forward(self, x): # slot shape [b*n,c], x shape [b*n,c,h,w]
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        return self.upsample(self.relu(out + x))


class SynthesisBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 inplanes,
                 outplanes,
                 num_layer,
                 upsample=False):
        super(SynthesisBlock, self).__init__()
        self.attn = SimplexAttention(outplanes, inplanes, max(outplanes//64, 1), 0)
        self.forward_layers = nn.ModuleList()
        for i in range(num_layer):
            self.forward_layers.append(ResLayer(outplanes, upsample=False if i < num_layer-1 else upsample))

    def forward(self, x, slots): # slot shape [b,k,c], x shape [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2,3).permute(0,2,1)
        attn, x = self.attn(slots, x, return_attn=True)
        x = x.permute(0,2,1).reshape(b,c,h,w)
        for i in range(len(self.forward_layers)):
            x = self.forward_layers[i](x)
        return x, attn


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

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class Generator(nn.Module):
    def __init__(self, slot_dim=512, base_dim=512, out_dim=3, base_resolution=7, target_resolution=224, block_num=4):
        super().__init__()
        self.grid = nn.Parameter(torch.randn([1,base_dim, base_resolution,base_resolution]), requires_grad=True) # [1,4,h,w]
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            upsample = base_resolution < target_resolution
            self.generator_blocks.append(SynthesisBlock(slot_dim, base_dim, 2, upsample=upsample))
            base_dim = base_dim//2 if upsample else base_dim
            base_resolution = base_resolution * 2 if upsample else base_resolution

        self.end_cnn = nn.Sequential(
            ResLayer(base_dim, False),
            ResLayer(base_dim, True),
            ResLayer(base_dim//2, False),
            nn.Conv2d(base_dim//2, out_dim, 1, 1, 0))
        
        # self.end_cnn = nn.Sequential(
        #     ResLayer(base_dim, False),
        #     ResLayer(base_dim, False),
        #     nn.Conv2d(base_dim, out_dim, 1, 1, 0))
        

    def forward(self, slots):
        b = slots.shape[0]
        init_grid = torch.repeat_interleave(self.grid,b,0)
        for i in range(len(self.generator_blocks)):
            init_grid, attn = self.generator_blocks[i](init_grid, slots)

        return self.end_cnn(init_grid), attn

        

if __name__ == '__main__':
    net = Generator()
    slot = torch.randn([1,11,512])
    y, _ = net(slot)
    print(y.shape)

    from thop import profile
    Flops, Params = profile(net,(slot,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))