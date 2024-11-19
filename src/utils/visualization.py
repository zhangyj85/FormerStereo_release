import numpy as np
import torch


def disp_color_func(disp):
    """
    Based on color histogram, convert the gray disp into color disp map.
    The histogram consists of 7 bins, value of each is e.g. [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    Accumulate each bin, named cbins, and scale it to [0,1], e.g. [0.114, 0.299, 0.413, 0.587, 0.701, 0.886, 1.0]
    For each value in disp, we have to find which bin it belongs to
    Therefore, we have to compare it with every value in cbins
    Finally, we have to get the ratio of it accounts for the bin, and then we can interpolate it with the histogram map
    For example, 0.780 belongs to the 5th bin, the ratio is (0.780-0.701)/0.114,
    then we can interpolate it into 3 channel with the 5th [0, 1, 0] and 6th [0, 1, 1] channel-map
    Inputs:
        disp: numpy array, disparity gray map in (Height * Width, 1) layout, value range [0,1]
    Outputs:
        disp: numpy array, disparity color map in (Height * Width, 3) layout, value range [0,1]
    """
    map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])
    # grab the last element of each column and convert into float type, e.g. 114 -> 114.0
    # the final result: [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    bins = map[0:map.shape[0] - 1, map.shape[1] - 1].astype(float)

    # reshape the bins from [7] into [7,1]
    bins = bins.reshape((bins.shape[0], 1))

    # accumulate element in bins, and get [114.0, 299.0, 413.0, 587.0, 701.0, 886.0, 1000.0]
    cbins = np.cumsum(bins)

    # divide the last element in cbins, e.g. 1000.0
    bins = bins / cbins[cbins.shape[0] - 1]

    # divide the last element of cbins, e.g. 1000.0, and reshape it, final shape [6,1]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    cbins = cbins.reshape((cbins.shape[0], 1))

    # transpose disp array, and repeat disp 6 times in axis-0, 1 times in axis-1, final shape=[6, Height*Width]
    ind = np.tile(disp.T, (6, 1))
    tmp = np.tile(cbins, (1, disp.size))

    # get the number of disp's elements bigger than  each value in cbins, and sum up the 6 numbers
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis=0)

    bins = 1 / bins

    # add an element 0 ahead of cbins, [0, cbins]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))
    cbins[1:] = t

    # get the ratio and interpolate it
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s, 0:3] * np.tile(1 - disp, (1, 3)) + map[s + 1, 0:3] * np.tile(disp, (1, 3))

    return disp


def gen_disp_error_colormap():
    """
    format:[[min_error<= this_error<=max_error, RGB],
            [min_error<= this_error<=max_error, RGB],
            ...
            [min_error<= this_error<=max_error, RGB],]
    unit: pixel
    """
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols

error_colormap = gen_disp_error_colormap()

def disp_error_image_func(D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1):
    """
    D_est_tensor: estimated disparity, BHW
    D_gt_tensor:  ground-truth disparity, BHW
    abs_thres: abs(D_est - D_gt) > abs_thres? outliers : zeros
    """
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    B, H, W = D_gt_np.shape

    # valid mask
    mask = D_gt_np > 1e-3

    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0     # remove the invalid pixels
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)

    # get colormap
    cols = error_colormap

    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]  # find out the correspondant range, give RGB values

    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));

    # remove the color in invalid pixels
    error_image[np.logical_not(mask)] = 0.

    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))



def gen_dep_error_colormap(max_range=2):
    """
    format:[[min_error<= this_error<=max_error, RGB],
            [min_error<= this_error<=max_error, RGB],
            ...
            [min_error<= this_error<=max_error, RGB],]
    unit: meter
    range: (0, max_range) meters
    """
    max_range = max_range / 0.9
    cols = np.array(
        [[0.0 * max_range, 0.1 * max_range, 49, 54, 149],
         [0.1 * max_range, 0.2 * max_range, 69, 117, 180],
         [0.2 * max_range, 0.3 * max_range, 116, 173, 209],
         [0.3 * max_range, 0.4 * max_range, 171, 217, 233],
         [0.4 * max_range, 0.5 * max_range, 224, 243, 248],
         [0.5 * max_range, 0.6 * max_range, 254, 224, 144],
         [0.6 * max_range, 0.7 * max_range, 253, 174, 97],
         [0.7 * max_range, 0.8 * max_range, 244, 109, 67],
         [0.8 * max_range, 0.9 * max_range, 215, 48, 39],
         [0.9 * max_range, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols

dep_error_colormap = gen_dep_error_colormap(max_range=2.)

def dep_error_image_func(D_est_tensor, D_gt_tensor):
    """
    D_est_tensor: estimated depth, BHW
    D_gt_tensor:  ground-truth depth, BHW
    abs_thres: abs(D_est - D_gt) > abs_thres? outliers : zeros
    """
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    B, H, W = D_gt_np.shape

    # valid mask
    mask = D_gt_np > 1e-3

    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0     # remove the invalid pixels

    # get colormap
    cols = error_colormap

    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]  # find out the correspondant range, give RGB values

    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));

    # remove the color in invalid pixels
    error_image[np.logical_not(mask)] = 0.

    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))