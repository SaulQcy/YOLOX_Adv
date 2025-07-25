import PIL.Image
from torchvision.transforms.transforms import Compose, ToTensor, Resize, Normalize, RandomCrop, RandomHorizontalFlip
from torch.utils.data.dataloader import DataLoader
import torchvision
# from tools.inject_backdoor import BadTransform
import random
import torch
from torch.utils.data import Dataset
import PIL

import os
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderWithoutSubdirs(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.img_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_dataloader(dataset_name: str, batch_size: int, pin_memory: bool, num_workers: int):
    train_ds, test_ds = get_train_and_test_dataset(dataset_name)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=pin_memory)
    return train_dl, test_dl

class List2Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        return x, y


def get_dataset_normalization(dataset_name):
    # this function is from BackdoorBench
    # given name, return the default normalization of images in the dataset
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif dataset_name in ["gtsrb", "celeba", 'fer2013', 'rafdb']:
        dataset_normalization = Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'cifar100':
        dataset_normalization = Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    elif dataset_name == "mnist":
        dataset_normalization = Normalize([0.5], [0.5])
    elif dataset_name == 'tiny':
        dataset_normalization = Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif dataset_name == 'imagenet':
        dataset_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset_name == 'imagenette':
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        dataset_normalization = Normalize(mean=[0.4648, 0.4543, 0.4247], std=[0.2785, 0.2735, 0.2944])
    else:
        raise NotImplementedError(dataset_name)
    return dataset_normalization

class DeNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, tensor):
        mean = self.mean.to(tensor.device)[None, :, None, None]
        std = self.std.to(tensor.device)[None, :, None, None]
        return tensor * std + mean

def get_de_normalization(dataset_name):
    if dataset_name == "cifar10":
        dataset_de_normalization = DeNormalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif dataset_name in ["gtsrb", "celeba", 'fer2013', 'rafdb']:
        dataset_de_normalization = DeNormalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'cifar100':
        dataset_de_normalization = DeNormalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    elif dataset_name == "mnist":
        dataset_de_normalization = DeNormalize([0.5], [0.5])
    elif dataset_name == 'tiny':
        dataset_de_normalization = DeNormalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif dataset_name == 'imagenet':
        dataset_de_normalization = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset_name == 'imagenette':
        dataset_de_normalization = DeNormalize(mean=[0.4648, 0.4543, 0.4247], std=[0.2785, 0.2735, 0.2944])
    else:
        raise NotImplementedError(dataset_name)
    return dataset_de_normalization


def get_benign_transform(dataset_name, train=True, random_crop_padding=4):
    _, size = get_dataset_class_and_scale(dataset_name)
    trans_list = [Resize((size, size))]
    if train:
        trans_list.append(RandomCrop((size, size), padding=random_crop_padding))
        if dataset_name == 'cifar10':
            trans_list.append(RandomHorizontalFlip())
    trans_list.append(ToTensor())
    trans_list.append(get_dataset_normalization(dataset_name))
    return Compose(trans_list)

def get_poison_transform(config, train=True, random_crop_padding=4):
    dataset_name = config.dataset_name
    _, size = get_dataset_class_and_scale(dataset_name)
    trans_list = [Resize((size, size))]
    if train:
        trans_list.append(RandomCrop((size, size), padding=random_crop_padding))
        if dataset_name == 'cifar10':
            trans_list.append(RandomHorizontalFlip())
    trans_list.append(ToTensor())
    trans_list.append(get_dataset_normalization(dataset_name))
    trans_list.append(BadTransform(config))
    return Compose(trans_list)


from omegaconf import DictConfig
class PoisonDataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.transform = BadTransform(config)
        self.do_norm = get_dataset_normalization(config.dataset_name)
        self.de_norm = get_de_normalization(config.dataset_name)
        self.config: DictConfig= config
        self.poisoned_num = 0

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if (random.random() < self.config.ratio) and self.config.attack.name != 'benign':
            # enhance
            if self.config.attack.mode == "train" and self.config.enhance != 0 and random.random() < 0.1:
                if self.config.attack.name == 'badnet':
                    _, h, w = x.shape
                    mask = PIL.Image.open(f'{self.config.attack.tg_path}/mask_{h}_{int(h / 10)}.png')
                    mask = mask.resize((h, w))
                    mask = ToTensor()(mask)
                    x = self.de_norm(x).squeeze()
                    x_p = self.transform(x)
                    x_p.clip_(0, 1)
                    x_e = self.do_norm(x_p) + self.config.enhance * torch.rand_like(x_p, device=x_p.device) * mask
                else:
                    x = self.de_norm(x).squeeze()
                    x_p = self.transform(x)
                    x_p.clip_(0, 1)
                    x_e = self.do_norm(x_p) + self.config.enhance * torch.rand_like(x_p, device=x_p.device)
                return x_e, y            
            # poisoned
            x = self.de_norm(x).squeeze()
            x_p = self.transform(x)
            x_p.clip_(0, 1)
            x_p = self.do_norm(x_p)
            x = x_p
            y = y - y + self.config.target_label
        return x, y

    def __len__(self):
        return len(self.dataset)

class PartialDataset(Dataset):
    def __init__(self, dataset, partial_ratio):
        self.dataset = dataset
        self.size = int(len(dataset) * partial_ratio)
        self.indices = random.sample(range(len(dataset)), self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_dataset_class_and_scale(dataset_name):
    if dataset_name == 'imagenette':
        scale = 224
        num_classes = 10
    elif dataset_name == 'cifar10':
        num_classes = 10
        scale = 32
    elif dataset_name == 'gtsrb':
        num_classes = 43
        scale = 32
    elif dataset_name == 'fer2013':
        num_classes = 8
        scale = 64
    elif dataset_name == 'rafdb':
        num_classes = 7
        scale = 64
    elif dataset_name == 'celeba':
        num_classes = 2
        scale = 128
    else:
        raise NotImplementedError(dataset_name)
    return num_classes, scale   


def get_train_and_test_dataset(dataset_name):
    if dataset_name == 'imagenette':
        train_ds = torchvision.datasets.Imagenette(root=DATA_PATH, split='train', transform=get_benign_transform(dataset_name))
        test_ds = torchvision.datasets.Imagenette(root=DATA_PATH, split='val', transform=get_benign_transform(dataset_name, train=False))
    elif dataset_name == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, transform=get_benign_transform(dataset_name), download=True)
        test_ds = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, transform=get_benign_transform(dataset_name, train=False), download=True)
    elif dataset_name == 'gtsrb':
        train_ds = torchvision.datasets.GTSRB(root=DATA_PATH, split='train', transform=get_benign_transform(dataset_name), download=True)
        test_ds = torchvision.datasets.GTSRB(root=DATA_PATH, split='test', transform=get_benign_transform(dataset_name, train=False), download=True)
    elif dataset_name == "celeba":
        def celeba_target_transform(target):
            gender_label = target[20]
            return gender_label 
        train_ds = torchvision.datasets.CelebA(root=DATA_PATH, split="train", download=False, transform=get_benign_transform(dataset_name), target_transform=celeba_target_transform)
        test_ds = torchvision.datasets.CelebA(root=DATA_PATH, split="test", download=False, transform=get_benign_transform(dataset_name), target_transform=celeba_target_transform)
        # val_ds = torchvision.datasets.CelebA(root=DATA_PATH, split="valid", download=False)
    else:
        raise NotImplementedError(dataset_name)
    return train_ds, test_ds

def clip_normalized_tensor(tensor, normalization):
    mean = torch.tensor(normalization.mean, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(normalization.std, device=tensor.device).view(1, -1, 1, 1)
    min_val = (0 - mean) / std
    max_val = (1 - mean) / std
    # print(min_val.shape)
    t = tensor.clamp(min=min_val, max=max_val)
    if len(t.shape) > 3 and t.shape[0] == 1:
        t = t.squeeze()
    return t


if __name__ == '__main__':
    dl, test_dl = get_dataloader('celeba', 16, False, 4)
    for batch, label in dl:
        print(label)
        break
    import matplotlib.pyplot as plt
    from tools.img import tensor2ndarray
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(tensor2ndarray(batch[0]))
    ax[-1].imshow(tensor2ndarray(batch[-1]))
    plt.show()