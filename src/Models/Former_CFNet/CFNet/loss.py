import torch.nn as nn
import torch.nn.functional as F
from Models.FormerStereo.loss import stereo_infoNCE


def model_loss(disp_ests, disp_gt, mask):
    if mask.float().mean() < 0.25:
        # 当前无效信息大于有效信息, 则直接返回 0.
        return 0. * disp_ests[0].mean()
    weights = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7, 2 * 1.0] #0.5,1,2
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        disp_est = disp_est.unsqueeze(1)    # BHW -> B1HW
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


class loss_func(nn.Module):

    def __init__(self, config):
        super(loss_func, self).__init__()
        self.max_disp = max(256, config['model']['max_disp'])   # CFNet 最小视差范围 256
        self.min_disp = config['model']['min_disp']

    def forward(self, data_batch, training_output):
        # target: B1HW, (min_disp, max_disp)
        # output: a dict containing multi-scale disp map
        disp_true = data_batch["gt1"].to('cuda', non_blocking=True)
        mask = (self.min_disp < disp_true) & (disp_true < self.max_disp)
        mask.detach_()
        loss = model_loss(training_output['training_output'], disp_true, mask)

        if "init_cost_volume" in training_output.keys():
            loss = 0.1 * loss
            for cost in training_output['init_cost_volume']:
                loss += stereo_infoNCE(cost, disp_true, mask)

        if "recon_loss" in training_output.keys():
            loss += training_output['recon_loss']

        # dist_true = disp2distribute(disp_true, self.max_disp, b=1)   # BDHW
        # celoss = CEloss(training_output['prob_output'], dist_true, mask)
        # loss = 0.1 * loss + celoss

        return loss
