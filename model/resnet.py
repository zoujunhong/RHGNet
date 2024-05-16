import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1):
        super(BasicBlock, self).__init__()
        self.norm1 = nn.GroupNorm(1, planes)
        self.norm2 = nn.GroupNorm(1, planes)

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(inplanes, planes, 3, stride, 1) if stride > 1 else nn.Identity()

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
    }

    def __init__(self,
                 depth=18,
                 in_channels=3,
                 stem_channels=64,
                 stem_stride=4,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 multi_scale_aggregation=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        assert len(strides) == num_stages
        
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, stem_stride, stem_stride, 0),
                nn.GroupNorm(1, stem_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels, stem_channels, 5, 1, 2),
                nn.GroupNorm(1, stem_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels, stem_channels, 5, 1, 2),
                nn.GroupNorm(1, stem_channels),
                nn.ReLU(inplace=True))

        self.res_layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            layer = nn.Sequential()
            for j in range(num_blocks):
                layer.add_module('stage_{}_block_{}'.format(i,j), 
                    BasicBlock(
                        inplanes=stem_channels if j == 0 else base_channels,
                        planes=base_channels,
                        stride=stride if j == 0 else 1))
            
            self.res_layers.append(layer)

            stem_channels = base_channels
            base_channels *= 2

        self.multi_scale_aggregation = multi_scale_aggregation
        if multi_scale_aggregation:
            self.head_layer = IterativeHead(in_channels=[base_channels//16,base_channels//8,base_channels//4,base_channels//2])

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)
        
        outs = []
        for i in range(len(self.res_layers)):
            res_layer = self.res_layers[i]
            x = res_layer(x)
            outs.append(x)
        
        if self.multi_scale_aggregation:
            outs = self.head_layer(outs)
        return outs


class IterativeHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.projects = nn.ModuleList()
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i == num_branchs-1:
                self.projects.append(
                    nn.Sequential(
                        nn.Conv2d(self.in_channels[i], self.in_channels[i], 3, 1, 1),
                        nn.GroupNorm(1, self.in_channels[i]),
                        nn.ReLU(),
                        nn.Conv2d(self.in_channels[i], self.in_channels[i], 1, 1, 0),)
                    )
            else:
                self.projects.append(
                    nn.Sequential(   
                        nn.Conv2d(self.in_channels[i], self.in_channels[i+1], kernel_size=3, stride=1, padding=1),          
                        nn.GroupNorm(1, self.in_channels[i+1]),   
                        nn.ReLU(),
                        nn.Conv2d(self.in_channels[i+1], self.in_channels[i+1], kernel_size=1, stride=1, padding=0)))

    def forward(self, x):
        x = x[::-1]
        last_x = None
        for i in range(len(x)):
            if last_x is None:
                last_x = x[i]
            else:
                last_x = x[i] + F.interpolate(last_x, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
            last_x = self.projects[i](last_x)
            
        return last_x





if __name__ == '__main__':
    model = ResNet(depth=34)
    x = torch.randn([1,3,224,224])
    from thop import profile
    Flops, Params = profile(model,(x,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))