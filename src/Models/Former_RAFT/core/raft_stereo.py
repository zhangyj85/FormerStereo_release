import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicMultiUpdateBlock
from .extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from .corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from .utils.utils import coords_grid, upflow8
from Models.FormerStereo.submodule import build_gwc_volume


class RAFT_CFG:
    def __init__(self):
        self.train_iters = 10                   # raft 的最终模型统一使用 22 iters
        self.valid_iters = 40
        self.corr_implementation = "reg"        # ["reg", "alt", "reg_cuda", "alt_cuda"]
        self.shared_backbone = False
        self.corr_levels = 4
        self.corr_radius = 4
        self.n_downsample = 2
        self.context_norm = "batch"             # ['group', 'batch', 'instance', 'none']
        self.slow_fast_gru = False
        self.n_gru_layers = 3
        self.hidden_dims = [128] * 3


class Former_RAFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        args = RAFT_CFG()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        # if args.shared_backbone:
        #     self.conv2 = nn.Sequential(
        #         ResidualBlock(128, 128, 'instance', stride=1),
        #         nn.Conv2d(128, 256, 3, padding=1))
        # else:
            # self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
        from Models.FormerStereo.vit_backbones import ViT_Dense
        self.fnet = ViT_Dense(backbone="dam_vitl14_518", features=[128] * 4, out_feats=256, readout="ignore", enable_attention_hooks=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
        # self.weight_load()
        self.fnet.weight_load()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)

    def renormalization(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, -1, 1, 1)
        img = (img * std + mean) * 255.
        return img

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        fmap1, fmap2, recon_left, recon_right = self.fnet(image1, image2)

        image1 = self.renormalization(image1)
        image2 = self.renormalization(image2)
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        self.freeze_bn()    # We keep BatchNorm frozen

        # run the context network
        # if self.args.shared_backbone:
        #     *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
        #     fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
        # else:
        cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]

        # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        iters = self.args.train_iters if self.training else self.args.valid_iters
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
            if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
            net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if not self.training and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(-1 * flow_up)

        output = {}
        output['disparity'] = flow_predictions[-1]

        if self.training:
            output['training_output'] = flow_predictions
            output['init_cost_volume'] = build_gwc_volume(fmap1, fmap2, image1.shape[-1] // 4, 1, "cosine")
            output['recon_loss'] = F.mse_loss(image1, recon_left)#1.0 * (F.mse_loss(image1, recon_left) + F.mse_loss(image2, recon_right))

        return output
