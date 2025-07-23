from adversarial_input_analyze.tools.img import tensor2ndarray, dct_2d_3c_full_scale
import numpy as np
import torch
from tqdm import tqdm
from adversarial_input_analyze.tools.utils import manual_seed
import numpy
import argparse
import matplotlib.pyplot as plt
from adversarial_input_analyze.tools.utils import rm_if_exist
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from adversarial_input_analyze.tools.dataset import ImageFolderWithoutSubdirs

def clip(data: numpy.ndarray) -> numpy.ndarray:
    if data.shape[0] > 64:
        return np.clip(a=data, a_min=1.5, a_max=4.5)
    else:
        from scipy.ndimage import gaussian_filter
        data = np.log1p(np.abs(data))
        data = gaussian_filter(data, sigma=2)
        return data
    
def dct_result(args):
    device = 'cpu' 
    bs = total = args.total
    is_clip=True

    # num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    # train_dl, test_dl = get_dataloader(config.dataset_name, total, config.pin_memory, config.num_workers)

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

    res_before = np.zeros((scale, scale, 3), dtype=np.float32)
    res_after = np.zeros((scale, scale, 3), dtype=np.float32)

    batch_benign = next(iter(benign_dl))
    batch_benign = batch_benign.to(device=device)

    batch_adv = next(iter(adv_dl))
    batch_adv = batch_adv.to(device=device)

    x_c4show = None
    x_p4show = None

    for i in tqdm(range(total)):
        x_space = batch_benign[i]  # this is a tensor
        x_space = tensor2ndarray(x_space)
        x_f = dct_2d_3c_full_scale(x_space.astype(float))
        res_before += x_f
        x_c4show = x_space
    res_before /= total

    for i in tqdm(range(total)):
        x_space = batch_adv[i]  # this is a tensor
        x_space = tensor2ndarray(x_space)
        x_f = dct_2d_3c_full_scale(x_space.astype(float))
        res_after += x_f
        x_p4show = x_space
    res_after /= total
    # plot_space_target_space(x_c4show, x_f, x_p4show, x_f_poison, is_clip=True)
    if is_clip:
        x_target = clip(res_before)
        x_process_target = clip(res_after)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(x_c4show)
    axs[0, 0].set_title(f'Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(x_target[:, :, 0], cmap='hot')
    axs[0, 1].set_title('Original Image DCT')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(x_p4show)
    axs[1, 0].set_title(f'after (attack) process')
    axs[1, 0].axis('off')

    im2 = axs[1, 1].imshow(x_process_target[:, :, 0], cmap='hot')
    axs[1, 1].set_title(f'(attacked) img in target space')
    axs[1, 1].axis('off')
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax)
    plt.tight_layout()
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
    dct_result(args)