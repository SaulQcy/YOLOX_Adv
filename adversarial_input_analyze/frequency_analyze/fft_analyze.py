import sys

sys.path.append('../../')
from adversarial_input_analyze.tools.img import tensor2ndarray, rgb2yuv, yuv2rgb, plot_space_target_space, dct_2d_3c_slide_window, dct_2d_3c_full_scale
from adversarial_input_analyze.tools.dataset import ImageFolderWithoutSubdirs, get_dataloader, get_de_normalization, get_dataset_class_and_scale
# from tools.inject_backdoor import patch_trigger
import numpy as np
import torch
from tqdm import tqdm
from adversarial_input_analyze.tools.img import fft_2d_3c, ifft_2d_3c
from adversarial_input_analyze.tools.img import ndarray2tensor
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import hydra
from omegaconf import DictConfig, OmegaConf
from adversarial_input_analyze.tools.utils import manual_seed
import random
import matplotlib.pyplot as plt
import os
import argparse
from adversarial_input_analyze.tools.utils import rm_if_exist
import PIL.Image
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader

def fft_result(args):
    bs = total = args.total
    device = 'cpu' 
    scale = 640
    trans = T.Compose([
        T.Resize((scale, scale)),
        T.ToTensor()
    ])
    benign_path = args.benign_path
    adv_path = args.adv_path

    benign_dataset = ImageFolderWithoutSubdirs(benign_path, transform=trans)
    adv_dataset = ImageFolderWithoutSubdirs(adv_path, transform=trans)

    benign_dl = DataLoader(benign_dataset, batch_size=bs, shuffle=False, num_workers=4)
    adv_dl = DataLoader(adv_dataset, batch_size=bs, shuffle=False, num_workers=4)

    amp_before = np.zeros((scale, scale, 3), dtype=np.float32)
    amp_after = np.zeros((scale, scale, 3), dtype=np.float32)
    pha_before = np.zeros((scale, scale, 3), dtype=np.float32)
    pha_after = np.zeros((scale, scale, 3), dtype=np.float32)

    batch_benign = next(iter(benign_dl))
    batch_benign = batch_benign.to(device=device)

    batch_adv = next(iter(adv_dl))
    batch_adv = batch_adv.to(device=device)

    x_c4show = None
    x_p4show = None

    for i in tqdm(range(total)):
        x_space = batch_benign[i]  # this is a tensor
        x_space = tensor2ndarray(x_space)
        x_fft = np.fft.fft2(x_space, axes=(0, 1))
        amp_c, pha_c = np.abs(x_fft), np.angle(x_fft)

        amp_before += amp_c
        pha_before += pha_c
        x_c4show = x_space

    for i in tqdm(range(total)):
        x_space = batch_adv[i]  # this is a tensor
        x_space = tensor2ndarray(x_space)
        x_fft = np.fft.fft2(x_space, axes=(0, 1))
        amp_c, pha_c = np.abs(x_fft), np.angle(x_fft)

        amp_after += amp_c
        pha_after += pha_c
        x_p4show = x_space

    amp_before /= total
    amp_after /= total
    pha_before /= total
    pha_after /= total

    amp_before = np.log1p(np.abs(amp_before)).astype(np.uint8)
    amp_after = np.log1p(np.abs(amp_after)).astype(np.uint8)

    amp_before = np.fft.fftshift(amp_before, axes=(0, 1))
    amp_after = np.fft.fftshift(amp_after, axes=(0, 1))

    pha_before = np.fft.fftshift(pha_before, axes=(0, 1))
    pha_after = np.fft.fftshift(pha_after, axes=(0, 1))

    _, ax = plt.subplots(2, 3, figsize=(15, 10))
    for axes in ax.flat:
        axes.set_axis_off()
    ax[0, 0].imshow(x_c4show)
    ax[0, 0].set_title('clean')
    ax[0, 1].imshow(amp_before[:, :, 0])
    ax[0, 1].set_title('clean amp')
    ax[0, 2].imshow(pha_before[:, :, 0])
    ax[0, 2].set_title('clean pha')

    ax[1, 0].imshow(x_p4show)
    ax[1, 0].set_title('poisoned')
    ax[1, 1].imshow(amp_after[:, :, 0])
    ax[1, 1].set_title('poisoned amp')
    ax[1, 2].imshow(pha_after[:, :, 0])
    ax[1, 2].set_title('poisoned pha')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--benign_path',
        type=str,
        default='/home/saul/proj/YOLOX/datasets/COCO2017/val2017'
    )
    parser.add_argument(
        '--adv_path',
        type=str,
        default='/home/saul/proj/YOLOX/datasets/COCO2017_adv/val2017'
    )
    parser.add_argument(
        '--total',
        type=int,
        default=128
    )
    args = parser.parse_args()
    fft_result(args)
    