# coding=gbk
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import torchvision
from torch.nn import functional as F
from utils.realdownsample.degradations import circular_lowpass_kernel, random_mixed_kernels,random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from utils.realdownsample.logger import get_root_logger
from utils.realdownsample.img_util import imfrombytes,img2tensor
from utils.realdownsample.img_process_util import filter2D,USMSharp
from utils.realdownsample.transforms import augment,paired_random_crop
from utils.realdownsample.DiffJPEG import DiffJPEG
import PIL.Image as Image

def kernel():
    #self.paths = img_paths
    #scale = 1 #缩放比例
    #gt_size = gt_size #输入图像的尺寸，需要h=w
    gpu_id = None #0
    device = None #
    if gpu_id:
        device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    # blur settings for the first degradation
    blur_kernel_size = 21
    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob =[0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # a list for each kernel probability
    sinc_prob = 0.1
    blur_sigma = [0.2, 3]
    betag_range = [0.5, 4] # betag used in generalized Gaussian blur kernels
    betap_range = [1, 2] # betap used in plateau blur kernels

    ## blur settings for the second degradation
    blur_kernel_size2 =21
    kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob2 = 0.1
    blur_sigma2 = [0.2, 1.5]
    betag_range2 = [0.5, 4]
    betap_range2 = [1, 2]

    # a final sinc filter
    final_sinc_prob = 0.8
    ############################
    use_hflip = False
    use_rot = False
    ################
    kernel_range = [2 * v + 1 for v in range(3, 11)] ## kernel size ranges from 7 to 21
    # TODO: kernel range is now hard-coded, should be in the configure file
    pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1

    file_client = None

    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma, [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list2,
            kernel_prob2,
            kernel_size,
            blur_sigma2,
            blur_sigma2, [-math.pi, math.pi],
            betag_range2,
            betap_range2,
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    # # BGR to RGB, HWC to CHW, numpy to tensor
    # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
    # img_gt = img_gt.unsqueeze(4)
    # print(img_gt.shape)
    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)

    return_d = {'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
    return return_d

def synthesis(gt):
    """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
    """
    gpu_id = None #0
    device = None #
    if gpu_id:
        device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    jpeger = DiffJPEG(differentiable=False).to(device) # # simulate JPEG compression artifacts
    usm_sharpener = USMSharp().to(device)  # do usm sharpening

    # the first degradation process
    resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
    resize_range = [0.7, 1.5]
    gaussian_noise_prob = 0.5
    noise_range = [1, 4]
    poisson_scale_range = [0.05, 1]
    gray_noise_prob = 0
    jpeg_range = [80, 95]

    # the second degradation process
    second_blur_prob = 0.8
    resize_prob2 = [0.2, 0.6, 0.2]  # up, down, keep
    resize_range2 = [0.7, 1.2]
    gaussian_noise_prob2 = 0.5
    noise_range2 = [1, 2]
    poisson_scale_range2 = [0.05, 0.5]
    gray_noise_prob2 = 0
    jpeg_range2 = [80, 95]

    # training data synthesis
    data = kernel()
    gt = gt.to(device)
    gt = gt.unsqueeze(0)
    gt_usm = usm_sharpener(gt)
    kernel1 = data['kernel1'].to(device)
    kernel2 = data['kernel2'].to(device)
    sinc_kernel = data['sinc_kernel'].to(device)

    ori_h, ori_w = gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(gt_usm, kernel1)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, resize_range[1])
    elif updown_type == 'down':
        scale = np.random.uniform(resize_range[0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = gray_noise_prob
    if np.random.uniform() < gaussian_noise_prob:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=poisson_scale_range,
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    mode1 = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / 2), int(ori_w / 2)),
        mode=mode1)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < second_blur_prob:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], resize_prob2)[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, resize_range2[1])
    elif updown_type == 'down':
        scale = np.random.uniform(resize_range2[0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h /2* scale * scale), int(ori_w /2* scale * scale)),
        mode=mode)
    # add noise
    gray_noise_prob = gray_noise_prob2
    if np.random.uniform() < gaussian_noise_prob2:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=poisson_scale_range2,
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        #out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        #out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
        out = filter2D(out, sinc_kernel)

    mode2 = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / 4), int(ori_w / 4)),
        mode=mode2)

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    # # random crop
    # gt_size = gt_size
    # (gt, gt_usm), lq = paired_random_crop([gt, gt_usm], lq, gt_size,
    #                                                      scale)

    # training pair pool
    #self._dequeue_and_enqueue()
    # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
    gt_usm = usm_sharpener(gt)
    lq = lq.contiguous()
    lq = lq.squeeze(0)
    return lq

import torchvision.transforms as transforms
import torchvision.utils as vutils


def load_image(image_path, size=(96, 96)):
    """
    加载并调整图片大小，然后转换为PyTorch张量。
    """
    # 打开图片并调整大小
    image = Image.open(image_path).convert('RGB')
    #image = image.resize(size)

    # 转换图片为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到[-1, 1]
    ])
    img_tensor = transform(image)
    img_tensor = (img_tensor + 1) / 2  # 从[-1, 1]转换到[0, 1]

    return img_tensor

from tqdm import tqdm
def process_images(input_folder, output_folder, size=(120, 120)):
    """
    处理指定文件夹中的所有图片，并保存到输出文件夹中。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构建完整的文件路径
            image_path = os.path.join(input_folder, filename)

            # 加载并处理图片
            img_tensor = load_image(image_path, size)

            # 假设有一个 synthesis 函数来处理图片
            result = synthesis(img_tensor)  # 这里假设 synthesis 是你定义的某个处理函数

            # 保存处理后的图片
            output_path = os.path.join(output_folder, filename)
            vutils.save_image(result, output_path)

if __name__ == '__main__':
    input_folder = r'../../datasets/train/HR'  # 替换为你的输入文件夹路径
    output_folder = r'../../datasets/train/HRdown'  # 替换为你的输出文件夹路径

    process_images(input_folder, output_folder)