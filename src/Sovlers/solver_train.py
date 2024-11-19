"""
Description:
    Project: xyz 
    This code is used for model training
    Author: Yongjian Zhang
    E-mail: zhangyj85@mail2.sysu.edu.cn
"""
import copy
import os
import sys
import time
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from Data import get_loader
from Models import MODEL, LOSS
from Metrics import METRICS

from utils.logger import ColorLogger
from utils.training_utils import *
import utils.distributed_utils as dist_utils
from utils.tools import *


class TrainSolver(object):
    def __init__(self, config):
        # 训练配置
        self.cfg = config
        self.reloaded = True if self.cfg['train']['resume'] is not None else False

        self.max_disp = self.cfg['model']['max_disp']
        self.min_disp = self.cfg['model']['min_disp']

        # 记录工具
        log_path = os.path.join('../logger', self.cfg['model']['name'])
        # self.writer = SummaryWriter(os.path.join(log_path, 'summary')) if dist_utils.is_main_process() else None
        self.logger = ColorLogger(log_path, 'logger.log')
        self.logger._print_configures(self.cfg)

        self.train_loader = get_loader(self.cfg)
        self.model = MODEL(self.cfg)
        self.loss_func = LOSS(self.cfg)
        self.metrics_func = METRICS(self.cfg)

        # 加载 cross-domain validation dataset
        from copy import deepcopy
        test_cfg = deepcopy(self.cfg)
        test_cfg["mode"] = "test"
        self.validation_loader = []
        self.validation_loader_name = []
        self.validation_loader_metric = []

        def add_val_set(dataset, split, metric):
            test_cfg["data"]["datasets"] = [dataset]
            test_cfg["split"] = split
            self.validation_loader.append(get_loader(test_cfg))
            self.validation_loader_name.append(dataset)
            self.validation_loader_metric.append(metric)

        # 手动更改验证集
        add_val_set(dataset="kitti2015", split="test", metric="D1-all")
        # add_val_set(dataset="kitti2012", split="test", metric="D1-all")
        add_val_set(dataset="middlebury", split="train-h", metric="Bad2.0")
        add_val_set(dataset="eth3d", split="test", metric="Bad1.0")

        # 梯度优化策略, 学习率调整
        steps_per_epoch = len(self.train_loader) if self.cfg['train']['eval_steps'] is None else self.cfg['train']['eval_steps']
        self.optimizer = create_optimizer(self.cfg['train'], self.logger, self.model)
        self.scheduler = create_scheduler(self.cfg['train'], self.logger, self.optimizer, steps_per_epoch=steps_per_epoch)
        self.fp16_scaler = None if not self.cfg['environment']['use_amp'] else torch.cuda.amp.GradScaler()

        # 控制训练迭代次数
        self.global_step = 0
        self.max_steps = self.cfg['train']['epoch'] * steps_per_epoch
        self.eval_steps = steps_per_epoch

        # 输出模型大小
        self.logger.info('Number of model parameters: {:.2f}M; Trainable: {:.2f}M'.format(
                sum([p.data.nelement() for p in self.model.parameters()]) / 1e6,
                sum([p.data.nelement() for p in self.model.parameters() if p.requires_grad]) / 1e6
            )
        )
        self.logger.info('TrainSet Size: {:.1f}k'.format(len(self.train_loader) / 1e3))

    def save_checkpoint(self):

        states = {
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        if self.fp16_scaler is not None:
            states['fp16_scaler'] = self.fp16_scaler.state_dict()

        ckpt_root = os.path.join(self.cfg['train']['save_path'], self.cfg['model']['name'])
        os.makedirs(ckpt_root, exist_ok=True)   # 创建保存路劲, 若存在则无操作
        ckpt_name = 'iter_{:d}.pth'.format(self.global_step)
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        torch.save(states, ckpt_full)
    
    def load_checkpoint(self):

        ckpt_full = self.cfg['train']['resume']
        # states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)
        states = torch.load(ckpt_full, map_location=torch.device('cpu'))

        # load params.
        if self.cfg['train']["strict"]:
            # 断点, 接着训练
            self.model.load_state_dict(states['model_state'], strict=True)
            self.optimizer.load_state_dict(states['optimizer_state'])
            self.scheduler.load_state_dict(states['scheduler_state'])
            self.global_step = states['global_step']
            if self.cfg['environment']['use_amp'] and "fp16_scaler" in states.keys():
                self.fp16_scaler.load_state_dict(states['fp16_scaler'])
        else:
            # 不强制要求网络层数和模型名称对应
            try:
                self.model.load_state_dict(states['model_state'], strict=True)
            except:
                for key in self.model.state_dict().keys():
                    if key not in states['model_state']:
                        print("Missing key:", key)
                print("Load model and ignore the missing keys.")
                self.model.load_state_dict(states['model_state'], strict=False)

    # def abnormal_estimation_check(self, disp_list):
    #     normal_flag = True
    #     for disp in disp_list:
    #         if torch.isnan(disp).any() or torch.isinf(disp).any():
    #             print("Abnormal estimations occur! Auto save the checkpoint.")
    #             self.save_checkpoint()
    #             normal_flag = False
    #             break
    #     return normal_flag

    def run(self):
        # move networks to gpu
        device = "cuda:{}".format(os.environ['LOCAL_RANK'])
        self.model = self.model.to(device)

        # 将 BN 换成 DDP 兼容的 Sync BN
        if dist_utils.has_batchnorms(self.model):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # 将学生模型分发到各个 GPU 上 (一个进程对应一个GPU)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[dist_utils.get_rank()],
            # find_unused_parameters=True
        )

        # 参数重载
        if self.reloaded:
            self.load_checkpoint()
            self.logger.info('[{:d}] Model loaded.'.format(self.global_step))
        else:
            self.logger.info('No pretrained model is used.')

        # 打印学习率
        print("The begin learning rate is {:.6f}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))

        # 对 train_loader 进行打乱
        self.train_loader.sampler.set_epoch(self.global_step // len(self.train_loader))
        data_iter = iter(self.train_loader)

        while True:

            # 达到最大训练轮数, 则停止训练
            if self.global_step >= self.max_steps:
                break

            # 获取数据
            try:
                data_batch = next(data_iter)
            except StopIteration:
                # 训练完一轮, 开启新的epoch
                self.train_loader.sampler.set_epoch(self.global_step // len(self.train_loader))
                data_iter = iter(self.train_loader)
                data_batch = next(data_iter)

            # 计算单次迭代训练时长
            torch.cuda.synchronize()
            start_time = time.time()

            # 清空梯度, 加载数据到 GPU, 并均分
            self.optimizer.zero_grad()
            data_batch_total = {}
            for key in data_batch.keys():
                if torch.is_tensor(data_batch[key]):
                    data_batch_total[key] = data_batch[key].to(device, non_blocking=True).chunk(self.cfg['train']['accumulate_grad_iters'])

            pred_total, used_total, loss_total = [], [], []
            # 将数据拆成 K 份
            for k in range(self.cfg['train']['accumulate_grad_iters']):
                # 对已有数据进行裁减, 得到学生的输入和监督信号
                self.model.train()                      # 模型训练, 加载数据
                imgL = data_batch_total["ir1"][k]
                imgR = data_batch_total["ir2"][k]
                label_dict = {
                    "gt1": data_batch_total["gt1"][k]
                }

                # 使用混合精度进行训练, 并计算损失
                sync_context = self.model.no_sync if (k + 1) % self.cfg['train']['accumulate_grad_iters'] != 0 else nullcontext
                with sync_context():
                    with torch.cuda.amp.autocast(self.fp16_scaler is not None, dtype=torch.float16):
                        training_output = self.model(imgL, imgR)
                        # if not self.abnormal_estimation_check(training_output['training_outputs']):
                        #     pickle_file = "../logger/error_iter_{:d}_GPU{}".format(self.global_step, dist_utils.get_rank())
                        #     save_variable(data_batch[k], pickle_file)
                        loss = self.loss_func(label_dict, training_output)
                        loss = loss / self.cfg['train']['accumulate_grad_iters']    # 梯度累加, 1/K 倍

                # 记录输出的结果
                pred_total.append(training_output['disparity'])
                used_total.append(data_batch_total["gt1"][k])
                loss_total.append(loss.data.cpu().item())

                # if torch.isinf(loss) or torch.isnan(loss):
                #     self.logger.warning("GPU {}:Invalid loss gradient occur! Skip this iteration.".format(dist_utils.get_rank()))
                #     if self.fp16_scaler is None:
                #         loss.backward()
                #     else:
                #         self.fp16_scaler.scale(loss).backward()
                #     self.optimizer.zero_grad()
                #     break   # 跳过当前迭代更新

                # 梯度反向传播 & 梯度累加 & 梯度裁减
                param_norms = None
                if self.fp16_scaler is None:
                    # 梯度反传
                    with sync_context():
                        loss.backward()
                    # # 梯度检查
                    # if check_grads(self.model):
                    #     self.logger.warning("GPU {}:Invalid loss gradient occur! Skip this iteration.".format(dist_utils.get_rank()))
                    # 梯度累加
                    if (k + 1) % self.cfg['train']['accumulate_grad_iters'] == 0:
                        # 梯度裁减
                        if self.cfg['environment']['clip_grad']:
                            param_norms = clip_gradients(self.model, self.cfg['environment']['clip_grad'])
                        if check_grads(self.model):
                            if check_weights(self.model):
                                self.logger.info("NAN weights occur in model!")
                                sys.exit(1)
                            else:
                                self.logger.info("NAN or INF grads exist, auto skips this iteration.")
                                self.optimizer.zero_grad()
                                break
                        self.optimizer.step()  # 参数更新
                else:
                    # 梯度反传
                    with sync_context():
                        self.fp16_scaler.scale(loss).backward()     # 使用同一个缩放因子对 loss 进行缩放, 因此可以累加
                    # 对当前梯度进行检查
                    # if check_grads(self.model):
                    #     self.logger.warning("GPU {}:Invalid loss gradient occur in iter:{}!".format(dist_utils.get_rank(), self.global_step))
                    # 梯度累加
                    if (k + 1) % self.cfg['train']['accumulate_grad_iters'] == 0:
                        # 梯度裁减
                        if self.cfg['environment']['clip_grad']:
                            self.fp16_scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                            param_norms = clip_gradients(self.model, self.cfg['environment']['clip_grad'])  # 已验证, 这种写法是有效的
                        # 参数更新前, 梯度检查
                        if check_grads(self.model):
                            if check_weights(self.model):
                                # 首先判断当前模型的权重是否出现 nan / inf
                                self.logger.info("NAN weights occur in model!")
                                sys.exit(1)
                            else:
                                # 数值溢出, 简单移除即可
                                self.logger.info("NAN or INF grads exist, auto skips this iteration and updates the scale factors.")
                        # scaler 更新参数, 会先自动 unscale 梯度, 如果有 nan 或 inf，自动跳过, 并更新 scale 缩放系数的大小, 减少下一轮溢出的可能
                        # 手动调用 unscale_() 后, scaler.step(optimizer)将不再自动反缩放
                        self.fp16_scaler.step(self.optimizer)
                        self.fp16_scaler.update()

            # 单次迭代时间计算
            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            # record the training info
            self.global_step += 1

            # 主进程输出当前迭代的具体信息
            if dist_utils.is_main_process():
                # 计算当前迭代的预测精度
                eval_metrics = self.metrics_func(
                    torch.cat(used_total, dim=0), torch.cat(pred_total, dim=0), training=True
                )
                # 利用 print 实现假的进度刷新
                print(
                    '[iter={:d}/{:6d} | lr={:.2f}e-4] Loss={:.2f}, EPE={:.2f}px, 3PE={:.2f}%, Time={:.2f}s.'.format(
                        self.global_step, self.max_steps, self.optimizer.state_dict()['param_groups'][0]['lr'] * 1e4,
                        sum(loss_total), eval_metrics["all-EPE"], eval_metrics["Bad3.0"] * 100, elapsed
                    ), end='\r'
                )

            # 学习率更新
            self.scheduler.step()

            # save and eval
            if (self.global_step % self.eval_steps == 0) or (self.global_step >= self.max_steps):

                if dist_utils.is_main_process():
                    # save checkpoint
                    self.save_checkpoint()
                    self.logger.info('')
                    self.logger.info('[{:d}] Model saved.'.format(self.global_step))

                    # cross-domain validation
                    self.model.eval()
                    val_metric_record = []
                    for val_loader, val_name, val_metric in zip(self.validation_loader, self.validation_loader_name, self.validation_loader_metric):
                        with torch.no_grad():
                            metric_sum = 0
                            N_total = 0
                            for val_batch in tqdm(val_loader, desc="rank {}".format(dist_utils.get_rank())):
                                imgL = val_batch["ir1"].to(device, non_blocking=True)
                                imgR = val_batch["ir2"].to(device, non_blocking=True)
                                disp = val_batch["gt1"].to(device, non_blocking=True)
                                if len(val_batch["mask1"]) > 0:
                                    ncc_mask_L = val_batch["mask1"].to(device, non_blocking=True)
                                    ncc_mask_L[ncc_mask_L < 0.999] = 0  # 参考 GraftNet & ITSA, 仅保留 I=255 的真值
                                else:
                                    ncc_mask_L = None

                                N_curr = imgL.shape[0]
                                output = self.model(imgL, imgR)
                                pred_disp = output["disparity"]

                                # 验证集的输入是一张完整的图像
                                eval_metrics = self.metrics_func(disp, pred_disp, training=False, ncc_mask=ncc_mask_L)
                                metric_sum += eval_metrics[val_metric] * N_curr
                                N_total += N_curr

                            # 单卡测试, 无需进行多卡数据收集
                            metric_avg = metric_sum / N_total
                            val_metric_record.append(metric_avg)

                        self.logger.info(
                            '[{:6d}/{:6d}] Validation : {} = {:.2f} %.'.format(
                                self.global_step, self.max_steps, val_name,
                                val_metric_record[-1] * 100
                            )
                        )
                    self.logger.info(
                        'The Next learning rate is {:.6f}'.format(self.optimizer.state_dict()['param_groups'][0]['lr'])
                    )
                    torch.cuda.empty_cache()    # 清空缓存
