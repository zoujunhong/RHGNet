import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint): 

    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros((B, npoint), dtype=torch.long).to(device)   # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10                       # 采样点到所有点距离（B, N）

    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch_size 数组
    
    barycenter = torch.mean(xyz, dim=1, keepdim=True)                   #计算重心坐标 及 距离重心最远的点

    dist = torch.sum((xyz - barycenter) ** 2, dim=-1)
    farthest = torch.argmax(dist, -1)                                   #将距离重心最远的点作为第一个点
    sampled_points = []
    for i in range(npoint):
        centroids[:, i] = farthest                                      # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)        # 取出这个最远点的xyz坐标
        sampled_points.append(centroid)
        dist = torch.sum((xyz - centroid) ** 2, -1)                     # 计算点集中的所有点到这个最远点的欧式距离
        distance[dist < distance] = dist[dist < distance]               # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离

        farthest = torch.argmax(distance, -1)                           # 返回最远点索引
    
    return torch.cat(sampled_points, dim=1)

class FFN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim))
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        return self.norm(x + self.encoder(x))
    
def QuietSoftmax(x, dim=-1):
    x = torch.exp(x)
    return x / (1 + torch.sum(x, dim=dim, keepdim=True))

def HardSoftmax(x, dim=-1):
    y_soft = F.softmax(x, dim=dim)
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(x).scatter_(dim, index, 1.)
    return (y_hard - y_soft).detach() + y_soft

############################################# Transformer #############################################
# -----------------------------------------------------------------------------------------------------

# Transpose tensor to scores
def transpose_for_scores(x, num_heads, elem_num, head_size):
    x = x.reshape(-1, elem_num, num_heads, head_size) # [B, N, H, S]
    x = x.permute(0, 2, 1, 3) # [B, H, N, S]
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob + 1e-8) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiheadAttention(torch.nn.Module):
    def __init__(self,
            output_cap_dim,           input_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .1,             # Attention dropout rate
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        self.to_q = nn.Linear(output_cap_dim, output_cap_dim)
        self.to_k = nn.Linear(input_cap_dim, output_cap_dim)
        self.to_v = nn.Linear(input_cap_dim, output_cap_dim)

        self.dim = output_cap_dim
        self.output_cap_dim = output_cap_dim
        self.to_dim = input_cap_dim
        
        self.num_heads = num_heads
        self.size_head = int(output_cap_dim / num_heads)

        self.norm = nn.LayerNorm(output_cap_dim) 
        self.dropout = DropPath(attention_dropout)

        self.proj = nn.Linear(output_cap_dim, output_cap_dim)


    def forward(self, input_cap, output_cap, mask=None, return_attn=False): # mask shape [B, input_num]
        # queries, keys and values
        i = self.norm(input_cap)
        o = self.norm(output_cap)
        queries = self.to_q(o)
        keys    = self.to_k(i)
        values  = self.to_v(i)
        # Reshape queries, keys and values, and then compute att_scores
        b, n1, c1 = input_cap.shape
        b, n2, c2 = output_cap.shape
        values  = transpose_for_scores(values,  self.num_heads, n1,   self.size_head)  # [B, N, T, H]
        queries = transpose_for_scores(queries, self.num_heads, n2,   self.size_head)  # [B, N, F, H]
        keys    = transpose_for_scores(keys,    self.num_heads, n1,   self.size_head)  # [B, N, T, H]
        att_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.size_head ** 0.5 # [B,N,output_num,input_num]
        if mask is not None:
            att_scores = att_scores + mask.unsqueeze(1).unsqueeze(1)
            
        att_probs = F.softmax(att_scores, dim=-1)

        # Compute weighted-sum of the values using the attention distribution
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        b, n, h, d = control.shape
        control = control.reshape(b, n, self.dim) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        output_cap = output_cap + self.dropout(self.proj(control))
        if return_attn:
            return att_probs, output_cap
        return output_cap


class SimplexAttention(torch.nn.Module):
    def __init__(self,
            output_dim,           
            input_dim,
            # Additional options
            num_heads           = 6,              # Number of attention heads
            attention_dropout   = 0,              # Attention dropout rate
        ):

        super().__init__()
        self.to_q = nn.Linear(output_dim, output_dim)
        self.to_k = nn.Linear(input_dim, output_dim)
        self.to_v = nn.Linear(input_dim, output_dim)

        self.dim = output_dim
        self.output_dim = output_dim
        self.to_dim = input_dim
        
        self.num_heads = num_heads
        self.size_head = int(output_dim / num_heads)

        self.norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        # self.dropout = DropPath(attention_dropout)

        self.modulation = nn.Sequential(
            nn.Linear(output_dim, 2*output_dim),
            nn.GELU(),
            nn.Linear(2*output_dim, 2*output_dim)
        )

        self.proj = nn.Linear(output_dim, output_dim)
    

    def integrate(self, tensor, control): # integration, norm
        # Normalize tensor
        tensor = self.norm(tensor)
        # Compute gain/bias
        control = self.modulation(control)
        gain, bias = torch.split(control, [self.output_dim, self.output_dim], dim = -1)
        tensor = tensor * (gain + 1) + bias
        return tensor


    def forward(self, input, output, mask=None, return_attn=False): # mask shape [B, input_num]
        # queries, keys and values
        queries = self.to_q(output)
        keys    = self.to_k(input)
        values  = self.to_v(input)
        # Reshape queries, keys and values, and then compute att_scores
        b, n1, c1 = input.shape
        b, n2, c2 = output.shape
        queries = transpose_for_scores(queries, self.num_heads, n2, self.size_head)  # [B, N, F, H]
        keys    = transpose_for_scores(keys,    self.num_heads, n1, self.size_head)  # [B, N, T, H]
        values  = transpose_for_scores(values,  self.num_heads, n1, self.size_head)  # [B, N, T, H]

        att_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.size_head ** 0.5 # [B,N,output_num,input_num]    
        att_probs = F.softmax(att_scores, dim=-1)
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        b, n, h, d = control.shape
        control = control.reshape(b, n, self.dim) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        output = self.integrate(output, self.proj(control))
        if return_attn:
            return att_probs, output
        return output


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self,
            output_cap_dim,           input_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .1,             # Attention dropout rate
            self_attn=True,
            attn_type='vanilla',
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        assert attn_type in ['vanilla', 'simplex']
        self.self_attn = self_attn
        if self_attn:
            self.self_attn = MultiheadAttention(output_cap_dim, output_cap_dim, num_heads, attention_dropout)
        self.multihead_attn = \
            MultiheadAttention(output_cap_dim, input_cap_dim, num_heads, attention_dropout) if attn_type == 'vanilla' \
            else SimplexAttention(output_cap_dim, input_cap_dim, num_heads, attention_dropout)
        self.droppath = DropPath(attention_dropout)
        self.FFN = nn.Sequential(
            nn.LayerNorm(output_cap_dim),
            nn.Linear(output_cap_dim,4*output_cap_dim),
            nn.GELU(),
            nn.Linear(4*output_cap_dim,output_cap_dim))

    def forward(self, input_cap, output_cap, mask_input=None, mask_output=None):
        if self.self_attn:
            output_cap = self.self_attn(output_cap, output_cap, mask=mask_output)
            
        attn, output_cap = self.multihead_attn(input_cap, output_cap, mask=mask_input, return_attn=True)
        output_cap = output_cap + self.droppath(self.FFN(output_cap))
        return attn, output_cap

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self,
            output_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .1             # Attention dropout rate
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        self.self_attn = MultiheadAttention(output_cap_dim, output_cap_dim, num_heads, attention_dropout)
        self.droppath = DropPath(attention_dropout)
        self.FFN = nn.Sequential(
            nn.LayerNorm(output_cap_dim),
            nn.Linear(output_cap_dim,4*output_cap_dim),
            nn.GELU(),
            nn.Linear(4*output_cap_dim,output_cap_dim)
        )


    def forward(self, input_cap, mask=None):
        input_cap = self.self_attn(input_cap, input_cap, mask=mask)
        input_cap = input_cap + self.droppath(self.FFN(input_cap))
        return input_cap