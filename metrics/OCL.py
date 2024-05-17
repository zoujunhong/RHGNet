import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def ARI(pred, gt, ignore = 100): # pred shape [h, w], gt shape [h, w]
    
    c1 = torch.max(gt) + 1
    c2 = torch.max(pred) + 1
    # print(c1,c2)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    
    valid = (gt != ignore).long()
    gt = gt[valid != 0]
    pred = pred[valid != 0]
    len = gt.shape[0]
    # print(len,c1,c2)
    with torch.no_grad():
        n = torch.zeros([len, c1*c2]).to(pred.device)
        index = (gt * c2 + pred).unsqueeze(-1).long()
        src = torch.ones([len,c1*c2]).to(pred.device)
        n.scatter_(1, index, src)

        n = n.reshape(len,c1,c2).sum(dim=0)
        
        a = torch.sum(n, dim=0)
        b = torch.sum(n, dim=1)

        RI = torch.sum(n * (n-1))
        ERI = torch.sum(a * (a-1)) * torch.sum(b * (b-1)) / (len * (len - 1))
        maxRI = 0.5 * (torch.sum(a * (a-1)) + torch.sum(b * (b-1)))

        ARI = (RI - ERI + 1e-8) / (maxRI - ERI + 1e-8)
        # if torch.isnan(ARI):
        #     print(RI, ERI, maxRI)
        return ARI


def ObjectIOU(pred, gt, ignore = 100): # pred shape [h, w], gt shape [h, w]
    
    temp = 1
    while True:
        if torch.sum(gt == temp) == 0:
            gt[gt > temp] -= 1
        elif torch.sum(gt == temp) > 0:
            temp += 1
        if temp > torch.max(gt):
            break
    
    c1 = torch.max(gt)
    c2 = torch.max(pred) + 1
    # print(c1,c2)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    
    valid = (gt != ignore).long()
    gt = gt[valid != 0]
    gt -= 1
    pred = pred[valid != 0]
    IOU = torch.zeros([max(c1,c2), max(c1,c2)]).to(pred.device)
    # print(c1,c2)
    for i in range(c1):
        for j in range(c2):
            I = torch.sum((gt == i) * (pred == j))
            U = torch.sum((gt == i) + (pred == j))
            IOU[i,j] = I / U
    # print(IOU)
    IOU_np = (IOU.cpu().numpy() * 100000).astype(np.int32)
    km = KMMatch(IOU_np)
    match = km.match()
    # print(match)
    IOU_list = []
    Pix_list = []
    OIOU = 0
    # print(match.shape)
    for i in range(match.shape[0]):
        OIOU += IOU[match[i],i]
        IOU_list.append(IOU[match[i],i].item())
        Pix_list.append(torch.sum(gt == match[i]).item())

    return OIOU/c1, c1, IOU_list, Pix_list

def MSC(pred, gt, ignore = 100): # pred shape [h, w], gt shape [h, w]
    temp = 1
    while True:
        if torch.sum(gt == temp) == 0:
            gt[gt > temp] -= 1
        elif torch.sum(gt == temp) > 0:
            temp += 1
        if temp > torch.max(gt):
            break
        
    c1 = torch.max(gt)+1
    c2 = torch.max(pred) + 1
    # print(c1,c2)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    
    valid = (gt != ignore).long()
    gt = gt[valid != 0]
    # gt -= 1
    pred = pred[valid != 0]
    IOU = torch.zeros([c1, c2]).to(pred.device)
    # print(c1,c2)
    count_0 = 0
    for i in range(1,c1):
        if torch.sum(gt == i) == 0:
            count_0+=1
        for j in range(c2):
            I = (gt == i) * (pred == j)
            I = torch.sum(I)
            U = (gt == i) + (pred == j)
            U = torch.sum(U)
            IOU[i,j] = I / (U+1e-8)
    IOU_max = torch.max(IOU, dim=-1)[0]
    MSC = torch.sum(IOU_max) / (c1-1-count_0)

    return MSC

def iou_binary(mask_A, mask_B):
    assert mask_A.shape == mask_B.shape
    assert mask_A.dtype == torch.bool
    assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())

def average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]
    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.
    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = torch.tensor(bsz*[0.0])
    N = torch.tensor(bsz*[0])
    scaled_scores = torch.tensor(bsz*[0.0])
    scaling_sum = torch.tensor(bsz*[0])

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0])
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_scores[N == 0] == 0).all()
    assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_scores[N == 0] == 0).all()
    assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension 
    return mean_sc.mean(0)


def QACriterionClevr(output, answers):
        loss = {}
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        # type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        # loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_binary = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_reg = answers["answer_type"] == 2

        binary_norm = is_binary.sum() if is_binary.any() else 1.0
        loss["loss_answer_binary"] = (
            F.binary_cross_entropy_with_logits(output["pred_answer_binary"], answers["answer_binary"], reduction="none")
            .masked_fill(~is_binary, 0)
            .sum()
            / binary_norm
        )

        reg_norm = is_reg.sum() if is_reg.any() else 1.0
        loss["loss_answer_reg"] = (
            F.cross_entropy(output["pred_answer_reg"], answers["answer_reg"], reduction="none")
            .masked_fill(~is_reg, 0)
            .sum()
            / reg_norm
        )

        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )

        loss['loss_total'] = loss["loss_answer_type"] + \
                            torch.mean(is_binary * loss["loss_answer_binary"] + is_reg * loss["loss_answer_reg"] + is_attr * loss["loss_answer_attr"])
        return loss

def safe_nll(pred, target, reduction='none', ignore_index=-100):
    pred = torch.clamp(pred, 1e-8, 1.)
    return F.nll_loss(torch.log(pred), target, reduction=reduction, ignore_index=ignore_index)

def QAFocus_Criterion_NLL(answers, ans_dict):
        question_pred, size, color, shape, material = ans_dict['type'], ans_dict['size_attention'], \
            ans_dict['color_attention'], ans_dict['shape_attention'], ans_dict['material_attention']

        loss = {}

        loss["loss_answer_type"] = safe_ce(question_pred, answers["answer_type"], reduction="mean")

        is_size = answers["answer_type"] == 0
        is_color = answers["answer_type"] == 1
        is_shape = answers["answer_type"] == 2
        is_material = answers["answer_type"] == 3
        loss["loss_answer_size"] = safe_nll(size, answers["answer_size"], reduction="none", ignore_index=255)

        loss["loss_answer_color"] = safe_nll(color, answers["answer_color"], reduction="none", ignore_index=255)

        loss["loss_answer_shape"] = safe_nll(shape, answers["answer_shape"], reduction="none", ignore_index=255)

        loss["loss_answer_material"] = safe_nll(material, answers["answer_material"], reduction="none", ignore_index=255)

        loss['loss_total'] = loss["loss_answer_type"] + \
                            torch.mean(is_size * loss["loss_answer_size"] + \
                                       is_color * loss["loss_answer_color"] + \
                                       is_shape * loss["loss_answer_shape"] + \
                                       is_material * loss["loss_answer_material"])
        return loss

def safe_ce(pred, target, reduction='none', ignore_index=-100):
    pred = F.softmax(pred, dim=-1)
    pred = torch.clamp(pred, 1e-8, 1.)
    return F.nll_loss(torch.log(pred), target, reduction=reduction, ignore_index=ignore_index)

def QAFocus_Criterion_Wholetask(answers, question_pred, size, color, shape, material, judge, count):
        loss = {}
        loss["loss_answer_type"] = safe_ce(question_pred, answers["answer_type"], reduction='mean')

        is_size = answers["answer_type"] == 0
        is_color = answers["answer_type"] == 1
        is_shape = answers["answer_type"] == 2
        is_material = answers["answer_type"] == 3
        is_judge = answers["answer_type"] == 4
        is_count = answers["answer_type"] == 5

        loss["loss_answer_size"] = safe_ce(size, answers["answer_size"], reduction='none')

        loss["loss_answer_color"] = safe_ce(color, answers["answer_color"], reduction='none')

        loss["loss_answer_shape"] = safe_ce(shape, answers["answer_shape"], reduction='none')

        loss["loss_answer_material"] = safe_ce(material, answers["answer_material"], reduction='none')

        loss["loss_answer_judge"] = safe_ce(judge, answers["answer_judge"], reduction='none')

        loss["loss_answer_count"] = safe_ce(count, answers["answer_count"], reduction='none')

        loss['loss_total'] = loss["loss_answer_type"] + \
                            torch.mean(is_size * loss["loss_answer_size"] + \
                                       is_color * loss["loss_answer_color"] + \
                                       is_shape * loss["loss_answer_shape"] + \
                                       is_material * loss["loss_answer_material"] + \
                                       is_judge * loss["loss_answer_judge"] + \
                                       is_count * loss["loss_answer_count"])
        return loss

def QAFocus_Criterion_CE(answers, question_pred, size, color, shape, material):
        loss = {}
        loss["loss_answer_type"] = F.cross_entropy(question_pred, answers["answer_type"])

        # type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        # loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_size = answers["answer_type"] == 0
        is_color = answers["answer_type"] == 1
        is_shape = answers["answer_type"] == 2
        is_material = answers["answer_type"] == 3

        loss["loss_answer_size"] = F.cross_entropy(size, answers["answer_size"], reduction="none", ignore_index=255)

        loss["loss_answer_color"] = F.cross_entropy(color, answers["answer_color"], reduction="none", ignore_index=255)

        loss["loss_answer_shape"] = F.cross_entropy(shape, answers["answer_shape"], reduction="none", ignore_index=255)

        loss["loss_answer_material"] = F.cross_entropy(material, answers["answer_material"], reduction="none", ignore_index=255)


        loss['loss_total'] = loss["loss_answer_type"] + \
                            torch.mean(is_size * loss["loss_answer_size"] + \
                                       is_color * loss["loss_answer_color"] + \
                                       is_shape * loss["loss_answer_shape"] + \
                                       is_material * loss["loss_answer_material"])
        return loss

def STEGO_Loss(feature, seg, threshold=0.2):
    # feature shape [b, c, h, w]
    # seg shape [b, n_class, h, w]
    b, c, h, w = feature.shape
    _, n, _, _ = seg.shape

    feature = feature.reshape(b,c,-1) 
    feature = feature / (torch.sum(feature**2, dim=1, keepdim=True)**0.5)
    seg = seg.reshape(b,n,-1) 
    # seg = seg / (torch.sum(seg**2, dim=1, keepdim=True)**0.5)

    feature = feature.permute(0,2,1).reshape(b//2,-1,c)
    seg = seg.permute(0,2,1).reshape(b//2,-1,n)
    
    feature_similarity = torch.matmul(feature, feature.permute(0,2,1))
    # _, hw, ij = feature_similarity.shape
    feature_similarity = feature_similarity - torch.mean(feature_similarity, dim=-1, keepdim=True)

    seg_similarity = torch.matmul(seg, seg.permute(0,2,1))

    feature_similarity = feature_similarity.reshape(-1)
    seg_similarity = seg_similarity.reshape(-1)

    loss = torch.mean(-(feature_similarity - threshold) * seg_similarity)

    return loss

# Kuhn-Munkres匹配算法
class KMMatch(object):

    def __init__(self, graph):
        assert isinstance(graph, np.ndarray), print("二分图的必须采用numpy array 格式")
        assert graph.ndim == 2, print("二分图的维度必须为2")
        self.graph = graph

        rows, cols = graph.shape
        self.rows = rows
        self.cols = cols

        self.lx = np.zeros(self.cols, dtype=np.float32)  # 横向结点的顶标
        self.ly = np.zeros(self.rows, dtype=np.float32)  # 竖向结点的顶标

        self.match_index = np.ones(cols, dtype=np.int32) * -1  # 横向结点匹配的竖向结点的index （默认-1，表示未匹配任何竖向结点）
        self.match_weight = 0  # 匹配边的权值之和

        self.inc = math.inf

    def match(self):
        # 初始化顶标, lx初始化为0，ly初始化为节点对应权值最大边的权值
        for y in range(self.rows):
            self.ly[y] = max(self.graph[y, :])

        for y in range(self.rows):  # 从每一竖向结点开始，寻找增广路
            while True:
                self.inc = np.inf
                self.vx = np.zeros(self.cols, dtype=np.int32)  # 横向结点的匹配标志
                self.vy = np.zeros(self.rows, dtype=np.int32)  # 竖向结点的匹配标志
                if self.dfs(y):
                    break
                else:
                    self.update()
                # print(y, self.lx, self.ly, self.vx, self.vy)
        return self.match_index

    # 更新顶标
    def update(self):
        for x in range(self.cols):
            if self.vx[x]:
                self.lx[x] += self.inc
        for y in range(self.rows):
            if self.vy[y]:
                self.ly[y] -= self.inc

    def dfs(self, y):  # 递归版深度优先搜索
        self.vy[y] = 1
        for x in range(self.cols):
            if self.vx[x] == 0:
                t = self.lx[x] + self.ly[y] - self.graph[y][x]
                if t == 0:
                    self.vx[x] = 1
                    # 两种情况：一是结点x没有匹配，那么找到一条增广路；二是X结点已经匹配，采用DFS，沿着X继续往下走，最后若以未匹配点结束，则也是一条增广路
                    if self.match_index[x] == -1 or self.dfs(self.match_index[x]):
                        self.match_index[x] = y  # 未匹配边变成匹配边
                        # print(y, x, self.match_index)
                        return True
                else:
                    if self.inc > t:
                        self.inc = t
        return False
# if __name__ == '__main__':
#     IOU = np.zeros((11,11))
#     IOU[:7,:] = np.random.rand(7,11)
#     print(IOU)
#     IOU_np = (IOU * 100000).astype(np.int32)
#     km = KMMatch(IOU_np)
#     match = km.match()
#     OIOU = 0
#     for i in range(match.shape[0]):
#         OIOU += IOU[match[i],i]
    
#     OIOU /= 7
#     print(OIOU)
#     # graph = np.array([[3,4,6,4,9],[6,4,5,3,8],[7,5,3,4,2],[6,3,2,2,5],[8,4,5,4,7]])
#     # km = KMMatch(graph)
#     # print(km.match())


if __name__ == '__main__':
    a = torch.randn([1,6,14,14])
    b = torch.randn([1,6,14,14])
    a = torch.argmax(a, dim=1, keepdim=True).squeeze()
    b = torch.argmax(b, dim=1, keepdim=True).squeeze()
    print(a.shape)
    print(ObjectIOU(a,a))
    # print(average_segcover(a,b)[0])
    # print(MSC(a.squeeze(),b.squeeze()))