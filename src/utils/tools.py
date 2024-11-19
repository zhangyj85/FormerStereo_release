import torch
import cv2
import os
import numpy as np
from .visualization import disp_color_func


class Dep2PcdTool(object):
    """docstring for Dep2PcdTool， 将深度图转为点云"""
    def __init__(self, valid_dep=1e-5,
                 mean = [0.485, 0.456, 0.406],
                 std  = [0.229, 0.224, 0.225]):
        super(Dep2PcdTool, self).__init__()
        self.valid_dep = valid_dep  # 最小有效深度的大小, 默认0.01mm
        self.mean = mean            # 3通道均值, 单通道图像需要重复输入
        self.std  = std             # 3通道标准差

    def renormalize(self, color, fprint=True):
        """
        input:  tensor, bchw, torch.normalize 之后的结果
        output: tensor, bhwc, (0,1)
        """
        mean = torch.tensor(self.mean)
        std  = torch.tensor(self.std)
        color = color * std[None, :, None, None].cuda() + mean[None, :, None, None].cuda()
        if fprint: print("Renormalize Image in shape [B, H, W, 3],", color.shape)
        return color.permute(0,2,3,1) * 255

    def colorize(self, dep, dep_max=None, dep_min=None, fprint=True):
        """
        input:  tensor, b1hw, (dmin, dmax)
        output: tensor, bhw3, (0-255)
        """
        mask = (dep > self.valid_dep).float()           # 深度小于 0.01mm, 认为是无效深度

        if dep_max == None: dep_max = dep.max()
        if dep_min == None: dep_min = dep[mask.bool()].min()

        # 有效深度内的归一化, 无效深度为0
        dep_norm = (dep - dep_min) / (dep_max - dep_min + 1e-8) * 255 * mask

        device = dep.device
        if device != 'cpu': dep_norm = dep_norm.to('cpu')

        dep_out = []
        for i in range(dep.shape[0]):
            dep_color = cv2.applyColorMap(cv2.convertScaleAbs(dep_norm[i].permute(1, 2, 0).data.numpy(), alpha=1.0), cv2.COLORMAP_JET)
            dep_color = dep_color * (mask[i].permute(1,2,0).data.cpu().numpy().astype(np.uint8))       # 无效深度的颜色是黑色
            dep_color = cv2.cvtColor(dep_color, cv2.COLOR_BGR2RGB)                  # opencv 默认格式为 BGR, 为了正常显示, 需要转换为 RGB
            dep_out.append(torch.tensor(dep_color))
        dep = torch.stack(dep_out)

        if fprint: print("Colorize Image in shape [B, H, W, 3],", dep.shape)
        return dep.to(device)

    def pcd2ply(self, rgb, dep, calib, ply_file):
        """
        dep: numpy array (H, W, 1), 0-100
        rgb: numpy array (H, W, 3), 0-255
        """
        rgb = np.array(rgb, dtype="float32")
        dep = np.array(dep, dtype="float32")
        pcd = self.rgbd2pcd(rgb, dep, calib)
        # f"{}" replace the contain with variable
        header = "ply\n" + \
                 "format ascii 1.0\n" + \
                 f"element vertex {pcd.shape[0]}\n" +\
                 "property float32 x\n" + \
                 "property float32 y\n" + \
                 "property float32 z\n" + \
                 "property uint8 red\n" + \
                 "property uint8 green\n" + \
                 "property uint8 blue\n" + \
                 "end_header\n"
        with open(ply_file, 'w+') as f:
            f.write(header)
            for i in range(pcd.shape[0]):
                x, y, z, r, g, b = pcd[i,:]
                line = '{:.5f} {:.5f} {:.5f} {:.0f} {:.0f} {:.0f}\n'.format(x,y,z,r,g,b)
                f.write(line)

    def rgbd2pcd(self, rgb, dep, calib):
        """
        rgb: numpy array (H, W, 3), (0,1)
        dep: numpy array (H, W, 1), (0,192)
        pcd: numpy array (N, 6)
        """
        xyz = self.dep2xyz(dep, calib)  # (N, 3), N=HW
        rgb = rgb.reshape(-1, 3)        # (N, 3)
        valid = (dep > self.valid_dep).reshape(-1, )
        pcd = np.concatenate([xyz, rgb], axis=1)
        pcd = pcd[valid, :]     # 仅保留有效深度的像素点
        return pcd                      # (N, 6)

    def dep2xyz(self, dep, calib):
        """
        dep: numpy.array (H, W, 1)
        cal: numpy.array (3, 3)
        xyz: numpy.array (N, 3)
        """
        # 参数输出以便确认
        print("Intrinsic:\n", calib)

        # 生成图像坐标
        u, v = np.meshgrid(np.arange(dep.shape[1]), np.arange(dep.shape[0]))    # (H, W, 2)
        u, v = u.reshape(-1), v.reshape(-1)                                     # (H*W,), (H*W,)

        # 构成所需的坐标矩阵
        img_coord = np.stack([u, v, np.ones_like(u)])   # (3, H*W), (u,v,1)
        cam_coord = np.linalg.inv(calib) @ img_coord    # (3,3)^(-1) * (3, HW)
        xyz_coord = cam_coord * dep[v, u, 0]            # (3, HW)
        return xyz_coord.T                              # (HW, 3)


class TensorImageTool(object):
    """ImageTool Function, 图像处理"""
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super(TensorImageTool, self).__init__()
        self.mean = torch.tensor(mean)
        self.std  = torch.tensor(std)

    def renormalize(self, color, fprint=True):
        """
        input:  tensor, bchw, torch.normalize 之后的结果
        output: tensor, bhwc, (0,255.)
        """
        mean = self.mean
        std  = self.std
        color = color * std[None, :, None, None].to(color.device) + mean[None, :, None, None].to(color.device)
        color = color.permute(0,2,3,1) * 255
        if fprint:
            print("Renormalize Image in shape [B, H, W, C],", color.shape, "data type:", color.dtype)
        return color

    def colorize(self, dep, dep_max=None, dep_min=None, fprint=True):
        """
        input:  tensor, b1hw, (dmin, dmax)
        output: tensor, bhw3, (0-255)
        """
        if dep_max == None:
            dep_max = np.percentile(dep[dep > 1e-3].data.cpu().numpy(), 97) # 取 97% 的最大值
        if dep_min == None:
            dep_min = np.percentile(dep[dep > 1e-3].data.cpu().numpy(), 3)  # 取  3% 的最小值

        dep_norm = dep#(dep - dep_min) / (dep_max - dep_min + 1e-8) * 255

        device = dep.device
        if device != 'cpu':
            dep_norm = dep_norm.to('cpu')

        dep_out = []
        for i in range(dep.shape[0]):
            # dmax * alpha + beta = 255
            # dmin * alpha + beta = 0
            alpha = 255.0 / (dep_max - dep_min)
            beta  = - dep_min * alpha
            dep_color = cv2.applyColorMap(cv2.convertScaleAbs(dep_norm[i].permute(1,2,0).data.numpy(), alpha=alpha, beta=beta), cv2.COLORMAP_JET)
            dep_out.append(torch.tensor(dep_color))
        dep = torch.stack(dep_out)

        if fprint:
            print("Colorize Image in shape [B, H, W, 3],", dep.shape)
        return dep.to(device)

    def colorize_disp(self, disp, disp_max=None, disp_min=None, fprint=True):
        """
        input:  tensor, b1hw, (dmin, dmax)
        output: tensor, bhw3, (0-255)
        """
        device = disp.device
        if device != 'cpu':
            disp = disp.to('cpu')

        color_disp = []
        for i in range(disp.shape[0]):
            temp_disp = disp[i]     # (1,H,W)
            # 对视差进行归一化
            if disp_max == None:
                disp_max = np.percentile(temp_disp[temp_disp > 1e-3].data.cpu().numpy(), 97) # 取 97% 的最大值
            if disp_min == None:
                disp_min = np.percentile(temp_disp[temp_disp > 1e-3].data.cpu().numpy(), 3)  # 取  3% 的最小值
            temp_disp = (temp_disp - disp_min) / (disp_max - disp_min)
            temp_disp = temp_disp.clamp(0, 1)

            # 视差图上色
            _, H, W = temp_disp.shape
            temp_disp = temp_disp.reshape(H*W, 1).data.numpy()
            color_temp_disp = disp_color_func(temp_disp)
            color_temp_disp = color_temp_disp.reshape(H, W, 3) * 255.
            color_disp.append(torch.from_numpy(color_temp_disp))
        color_disp = torch.stack(color_disp)

        if fprint:
            print("Colorize Image in shape [B, H, W, 3],", color_disp.shape)
        return color_disp.to(device)

    def ImageSave(self, img, save_path=None):
        """
        input: tensor, chw, color save, uint8
        """
        if img.device != 'cpu':
            img = img.to('cpu')
        img = img.data.numpy()   # np.array, HWC
        if save_path == None:
            save_path = './TempImage.png'


        # 注意, 如果使用 Image 打开(RGB), 需要对通道进行转换(RGB -> BGR)
        cv2.imwrite(save_path, img[..., ::-1])

    def DepthSave(self, depth, scale=1, save_path=None):
        """
        input: tensor, hw1, depth save, 16bit png
        需要再检查一下, 存在 0.1mm 的误差
        """
        if depth.device != 'cpu':
            depth = depth.to('cpu')
        depth = depth.data.numpy() / scale      # np.array, HWC
        depth = depth.astype(np.uint16)
        if save_path == None:
            save_path = './Temp-16bit.png'
        cv2.imwrite(save_path, depth)


def main():
    from PIL import Image
    import torchvision.transforms.functional as TF
    disp, _ = pfm_imread("/media/zhangyj85/Dataset/Stereo Datasets/Middlebury/2021/data/artroom1/disp0.pfm")
    disp = np.ascontiguousarray(disp, dtype=np.float32)
    disp = Image.fromarray(disp.astype('float32'), mode='F')
    K1 = np.array([[1733.74, 0, 792.27],
          [0, 1733.74, 541.89],
          [0,0,1]])
    K2 = np.array([[1733.74, 0, 792.27],
          [0, 1733.74, 541.89],
          [0,0,1]])
    baseline = 536.62
    dep = (K1[0, 2] - K2[0, 2]) + K1[0, 0] * baseline / (np.array(disp) + 1e-8)
    dep = Image.fromarray(dep)
    dep = TF.to_tensor(np.array(dep))
    dep = dep.unsqueeze(0) # tensor, B1HW
    PCDTool = Dep2PcdTool()
    PCDTool.pcd2ply(torch.ones_like(dep).repeat(1, 3, 1, 1)[0].data.cpu().numpy(), dep[0].permute(1, 2, 0).data.cpu().numpy(), K1, '/home/zhangyj85/Desktop/demo.ply')



def pfm_imread(filename):
    import re
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def plot_basis(feat, record=None):
    import torch.nn.functional as F
    from sklearn import decomposition
    feat = F.interpolate(feat, scale_factor=4, mode='bilinear')
    n, c, h, w = feat.shape
    n_components = 5
    feat = feat[0].squeeze().reshape(c, h*w).permute(1, 0)  # [hw,c]
    feat = feat.data.cpu().numpy()
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(feat)
    pca_features = pca.transform(feat).reshape(h, w, -1)
    norm_feat = np.sqrt(np.sum(pca_features ** 2, axis=2))
    if record is not None:
        os.makedirs(record, exist_ok=True)
        import matplotlib.pyplot as plt
        for i in range(n_components):
            plt.imsave(os.path.join(record, 'pca_fea_{}.svg'.format(i)), pca_features[:,:,i], cmap='jet')
        plt.imsave(os.path.join(record, 'pca_fea_sqrt.svg'), norm_feat, cmap='jet')
    return pca_features, norm_feat


if __name__ == "__main__":
    main()