import torch
import torch.nn as nn
import torch.nn.functional as F


def constractive_samples(cost, disp_gt, topk=4):
    # cost in shape (B,2C,D,H,W), where C is a normalized features
    # disp_gt is the full-size ground-truth in shape (B,1,h,w)
    # topk=4, 考虑了平坦区域的宽单峰分布, 以及边缘区域的双尖峰分布
    B, C, D, H, W = cost.shape
    _, _, h, w = disp_gt.shape
    if w != W:
        assert (h % H == 0) and (w % W == 0), "Invalid shape in calculating Stereo Info NCE Loss!"
        disp_gt = F.unfold(disp_gt, kernel_size=(h // H, w // W), stride=(h // H, w // W)).view(B, -1, 1, H, W)     # B,N,1,H,W
        disp_gt = disp_gt * (W / w)
    else:
        disp_gt = disp_gt.view(B, 1, 1, H, W)
    N = disp_gt.shape[1]        # disp_gt in shape (B,N,1,H,W)

    # map gt disp into volume shape
    disp_proposals = torch.arange(0, D, device=disp_gt.device, requires_grad=False).view(1, 1, -1, 1, 1).repeat(B, N, 1, H, W)    # B,N,D,H,W
    disp_volume = 1 - torch.abs(disp_gt - disp_proposals)       # (B,N,D,H,W), 视差到候选视差的距离, 越小越靠近真值. 1-距离得到权重
    disp_volume[disp_volume < 0] = 0.                           # 权重非负
    disp_volume = torch.sum(disp_volume, dim=1, keepdim=False)  # N 个真值赋予的权重进行求和, 得到每个候选视差的权重, (B,D,H,W)
    disp_volume = disp_volume / torch.sum(disp_volume, dim=1, keepdim=True)     # 将权重转化为概率, 分母等同于 N
    _, ind = disp_volume.sort(dim=1, descending=True)  # 降序排列, 则权重靠前的为正样本
    positive_ind = ind[:, :2, :, :]                         # (B,2,H,W), 仅考虑前两个作为正例, 因为第二个开始权重就已经锐减了
    negative_ind = ind[:, topk:, :, :]                     # (B,D-2,H,W)

    positive_samples = torch.gather(cost, dim=2, index=positive_ind.view(B, 1, 2, H, W).repeat(1, C, 1, 1, 1))
    negative_samples = torch.gather(cost, dim=2, index=negative_ind.view(B, 1, D - topk, H, W).repeat(1, C, 1, 1, 1))
    # negative_samples[negative_samples == 0] -= 1    # 将未进行相关的区域的相关系数置为-1，减轻对 loss 的影响 (仅对 corr volume 有效)

    return positive_samples, negative_samples


def get_occlusion_mask(disparity):
    # disparity in left view
    _, _, _, W = disparity.shape
    index = torch.arange(0, W, device=disparity.device, requires_grad=False).view(1, 1, 1, W)
    matching_right = index - disparity      # W_r = W_l - Disp
    # invisible region
    visible = (matching_right > 0.).float() # B1HW, 坐标为正
    # occlusion region
    count = 0.
    for i in range(1, W):
        shift_map = F.pad(matching_right[:, :, :, i:], (0, i, 0, 0), mode='constant', value=-1)     # shift in left
        count = count + (torch.abs(matching_right - shift_map) < 0.5).float()                       # 0.5 means round
    occlud = (count > 0.).float()   # 映射唯一
    # TODO: 增加形态学处理去除孔洞
    # 最终得到的不包含invisible和occlusion的mask
    valid_mask = visible * (1 - occlud)
    return valid_mask.bool().float().detach()


def stereo_infoNCE(cost, disp, mask=None, t=0.07, topk=4):
    # cost in shape (B,2C,D,H,W). 为了确保一致性, 避免发生错误, 要求输入的 cost volume 是 left & right 进行余弦相似性度量后的结果, 即 (B,1,D,H,W)
    # disp in shape (B,1,H,W)
    B, C_, D, H, W = cost.shape
    _, _, h, w = disp.shape

    temp_disp = F.interpolate(disp * W / w, size=[H, W], mode="nearest")
    if mask is not None:
        # mask取最小区域, 保证不会引入噪声监督信号
        mask = F.avg_pool2d(mask.float(), kernel_size=(h // H, w // W), stride=(h // H, w // W))
        mask = (mask > 0.75).float()
    else:
        mask = torch.ones_like(temp_disp)
    visible = get_occlusion_mask(temp_disp)
    mask = mask * visible
    if not mask.bool().any():
        # mask 全为 0, 返回 0
        return 0. * cost.mean()

    # calculate the similarity
    positive_samples, negative_samples = constractive_samples(cost, disp, topk=topk)
    if C_ >= 2:
        # 通道数大于1, 说明使用了 concat volume
        # TODO: 处理 concate 0 的情况
        C = C_ // 2
        p_score = F.cosine_similarity(positive_samples[:, :C, :, :, :], positive_samples[:, C:, :, :, :], dim=1)    # (B,2,H,W)
        n_score = F.cosine_similarity(negative_samples[:, :C, :, :, :], negative_samples[:, C:, :, :, :], dim=1)    # (B,D-2,H,W)
    else:
        # 通道数为1, 则输入为余弦相似度
        p_score = positive_samples.squeeze(1)   # (B,1,N,H,W) -> (B,N,H,W)
        n_score = negative_samples.squeeze(1)
    n_score[n_score == 0] = -1    # 由于构造 gwc volume 或者 concate volume 存在 0 padding, 将这些无效负样本的影响降到最小, 即将其相关性设为最小

    # info NCE loss
    lables = torch.zeros(B * H * W, dtype=torch.long, device=p_score.device)    # (B*H*W,1), indicate the 0-th is gt
    mask = mask.reshape(B * H * W)
    p_score_list = torch.chunk(p_score, chunks=p_score.shape[1], dim=1)     # 切分成 top-k 个正例
    infoNCE_loss = 0
    for i, p_score_sample in enumerate(p_score_list):
        # TODO: 考虑将 n_score 进行 detach, 解除负样本的梯度
        logits = torch.cat([p_score_sample, n_score], dim=1)        # (B, D-topk+1, H, W)
        logits = logits.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        loss = F.cross_entropy(logits / t, lables, reduction="none")
        loss = loss[mask.bool()].mean()                                     # 有效区域内进行平均
        infoNCE_loss += (0.5 ** i) * loss

    return infoNCE_loss


def disp2distribute(disp_gt, max_disp, b=2):
    # disp_gt & gt_distribute in shape (B1HW)
    disp_range = torch.arange(0, max_disp).view(1, -1, 1, 1).float().cuda()
    gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
    gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
    return gt_distribute


def CEloss(dist_ests, dist_gt, mask):
    weights = [1.0, 0.7, 0.5]
    all_losses = []
    for dist_est, weight in zip(dist_ests, weights):
        if dist_est.shape == 5:
            dist_est.squeeze(1)     # BDHW
        log_dist_est = torch.log(dist_est + 1e-8)   # avoid overflow
        ce_loss = torch.sum(-dist_gt * log_dist_est, dim=1, keepdim=True)   # B1HW loss
        ce_loss = torch.mean(ce_loss[mask])                                 #
        all_losses.append(weight * ce_loss)
    return sum(all_losses)
