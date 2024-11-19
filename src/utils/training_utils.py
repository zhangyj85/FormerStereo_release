import os
import math
import pickle

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
import utils.distributed_utils as dist_utils


def create_optimizer(config, logger, model):

    opt_name = config['optimizer_type'].lower()
    lr_max = config['lr_max']
    params = filter(lambda p: p.requires_grad, model.parameters())  # 带有梯度的参数参与优化, 否则为冻结的参数

    if opt_name == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr_max)
        logger.info('Use RMSProp optimizer, lr_max is {:.4f}'.format(lr_max))

    elif opt_name == 'adam':
        betas = (0.9, 0.999)    # 默认参数
        optimizer = optim.Adam(params, lr=config['lr_max'], betas=betas)
        logger.info('Use Adam optimizer, lr_max is {:.4f}, betas={:s}'.format(lr_max, str(betas)))

    elif opt_name == 'adamw':
        weight_decay = 0.01
        optimizer = optim.AdamW(params, lr=config['lr_max'], weight_decay=weight_decay)
        logger.info(['Use AdamW optimizer, lr_max is {:.4f}, weight_decay={:f}'.format(lr_max, weight_decay)])

    else:
        raise NotImplementedError('Optimizer type [{:s}] is not supported'.format(config['optimizer_type']))

    return optimizer


def create_scheduler(config, logger, optimizer, steps_per_epoch=1):

    shr_name = config['scheduler_type'].lower()

    if shr_name == 'multisteplr':
        # 当 global_step = milestones[i], 则 lr *= gamma
        milestones_epoch = [28, 34, 36, 38]     # 28轮后, lr = 3.75e-5
        milestones_step = [x * steps_per_epoch for x in milestones_epoch]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_step, gamma=1/2)
        logger.info('Use MultiStepLR scheduler, milestones of epoch is {:s}'.format(str(milestones_epoch)))

    elif shr_name == 'onecyclelr':
        # this scheduler is used for RAFT-based method, following raft-stereo
        max_lr = config['lr_max']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=config['epoch'] * steps_per_epoch + 100,
            pct_start=0.01,                          # warm up ratio, following RAFT
            cycle_momentum=False,
            anneal_strategy="linear",                  # {'cos', 'linear'}
        )
        logger.info('Use OneCycleLR scheduler')

    else:
        raise NotImplementedError('Scheduler type [{:s}] is not supported'.format(config['scheduler_type']))

    return scheduler


# def create_scheduler(config, niter_per_ep):
#     sch_name = config['scheduler_type'].lower()
#     epoch_num = config['epoch']
#     total_iter = epoch_num * niter_per_ep
#     if sch_name == 'constant':
#         scheduler = np.linspace(config['lr_max'], config['lr_max'], total_iter)
#     elif sch_name == 'cosine':
#         # 线性升温
#         warmup_iters = config['warmup_epoch'] * niter_per_ep
#         warmup_schedule = np.linspace((config['lr_max'] + config['lr_min']) / 2, config['lr_max'], warmup_iters)
#         # 余弦降温
#         iters = np.arange(epoch_num * niter_per_ep - warmup_iters)
#         schedule = [config['lr_min'] + 0.5 * (config['lr_max'] - config['lr_min']) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]
#         schedule = np.array(schedule)
#         scheduler = np.concatenate((warmup_schedule, schedule))
#     return scheduler


# def update_scheduler(optimizer, scheduler, global_step):
#     # TODO: update weight decay scheduler like ConvNeXt
#     for i, param_group in enumerate(optimizer.param_groups):
#         param_group["lr"] = scheduler[global_step - 1]


def loss_validation(loss, logger, data_batch, iter):
    # check for the invalid loss gradient, save the data and model parameters
    if torch.isinf(loss) or torch.isnan(loss):
        # 记录错误信息
        logger.info("GPU {}:Invalid loss gradient occur! Auto save the data and parameters.".format(dist_utils.get_rank()))
        # 保存当前数据
        errdir = "./logger/training"
        if not os.path.exists(errdir):
            os.makedirs(errdir)
        pickle_file = errdir + "/error_iter_{:d}_GPU{}".format(iter, dist_utils.get_rank())
        save_variable(data_batch, pickle_file)
        return False
    return True


def save_variable(val, filename):
    """Error Detection & Reload for Debug"""
    file = open(filename, 'wb')
    pickle.dump(val, file)
    file.close()


def load_variable(filename):
    file = open(filename, 'rb')
    val = pickle.load(file)
    file.close()
    return val


def check_grads(model):
    NAN_FLAG = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                # print("GPU {}: invalid gradient found, reset grad to zero.".format(dist_utils.get_rank()))
                NAN_FLAG = True
                break
    return NAN_FLAG


def check_weights(model):
    NAN_FLAG = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any() or torch.isinf(param).any():
                # print("GPU {}: invalid gradient found, reset grad to zero.".format(dist_utils.get_rank()))
                NAN_FLAG = True
                break
    return NAN_FLAG

"""*****************************************************************
*****************    SummaryWriter 保存标量/图像    ******************
*****************************************************************"""
def make_iterative_func(func):
    # 若 vars 为 list/dict/tuple, 对其元素逐个进行 warpper(x) 操作
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


def save_scalars(summary_writer, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = '{}/{}'.format(mode_tag, tag)
            # if len(values) > 1:
            scalar_name = scalar_name + "_" + str(idx)
            summary_writer.add_scalar(scalar_name, value, global_step)


def save_images(logger, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = '{}/{}'.format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True), global_step)


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms