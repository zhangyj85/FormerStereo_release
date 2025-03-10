import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.FormerStereo.loss import stereo_infoNCE


def model_loss(disp_ests, disp_gt, mask):
    if mask.float().mean() < 0.25:
        # 当前无效信息大于有效信息, 则直接返回 0.
        return 0. * disp_ests[0].mean()
    loss_gamma = 0.9
    n_predictions = len(disp_ests)
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
    weights = [adjusted_loss_gamma**(n_predictions - i - 1) for i in range(n_predictions)]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # assert not torch.isnan(disp_est).any() and not torch.isinf(disp_est).any()
        all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


class loss_func(nn.Module):

    def __init__(self, config):
        super(loss_func, self).__init__()
        self.max_disp = config['model']['max_disp']
        self.min_disp = config['model']['min_disp']

    def forward(self, data_batch, training_output):
        # target: B1HW, (min_disp, max_disp)
        # output: a dict containing multi-scale disp map
        disp_true = data_batch["gt1"].to('cuda', non_blocking=True)
        mask = (self.min_disp < disp_true) & (disp_true < self.max_disp)
        mask.detach_()
        loss = model_loss(training_output['training_output'], disp_true, mask)

        if "init_cost_volume" in training_output.keys():
            # 平衡视差 loss 和其他计量 loss 之间的权重
            loss = 0.1 * loss + stereo_infoNCE(training_output['init_cost_volume'], disp_true, mask)

        if "recon_loss" in training_output.keys():
            loss += training_output['recon_loss']

        return loss
