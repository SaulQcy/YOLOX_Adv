#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    # parser.add_argument(
    #     "-f",
    #     "--exp_file",
    #     default=None,
    #     type=str,
    #     help="please input your experiment description file",
    # )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='/home/saul/proj/YOLOX/exps/example/custom/dms.py',
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        '--adv',
        help='Whether conduct adversarial attack. 0 is not, 1 is PGD_Linf',
        default=0,
        type=int
    )

    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)


    model = exp.get_model()

    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.adv == 0:
        pass
    elif args.adv == 1:
        # PGD attack
        import cv2
        from tqdm import tqdm
        from PIL import Image
        from torchvision import transforms
        from adversarial_attack.PGD import pgd_attack
        from tools.os import rm_if_exist
        # PGD_Linf
        class Args:
            pass
        opt = Args()
        opt.__dict__ = {
            'attack': 'PGD_Linf',
            'N': 5,
            'sigma': 16.0,
            'momentum': 1.0,
            'num_iter': 4,
            'max_epsilon': 64.,
            'rho': 0.5,
            'bs': 8,
            'max_len': 2000,
        }
        # craft AEs
        dir_sc = f'{exp.data_dir}/val2017'
        dir_dst = f'{exp.data_dir}_adv/val2017'
        rm_if_exist(dir_dst)
        os.makedirs(dir_dst, exist_ok=False)
        img_names = [f for f in os.listdir(dir_sc) if f.endswith(('.jpg', '.png'))]
        model.eval().cuda()
        for name in tqdm(img_names, desc="Generating Adversarial Examples"):
            img_path = os.path.join(dir_sc, name)
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).cuda()  # shape: [1, C, H, W]
            ori_shape = (img_tensor.shape[-2], img_tensor.shape[-1])
            img_tensor = transforms.Resize((640, 640))(img_tensor)
            adv_tensor = pgd_attack(img_tensor, model, opt)  # shape: [1, C, H, W]
            adv_tensor = adv_tensor.squeeze(0).detach().cpu().clamp(torch.min(img_tensor).cpu(), torch.max(img_tensor).cpu())
            adv_tensor = transforms.Resize(ori_shape)(adv_tensor)
            adv_img = transforms.ToPILImage()(adv_tensor)
            adv_img.save(os.path.join(dir_dst, name))
        exp.data_dir = f'{exp.data_dir}_adv/'
    elif args.adv == 2:
        # random pixel perturbation.
        import cv2
        from tqdm import tqdm
        from PIL import Image
        from torchvision import transforms
        from adversarial_attack.random_pixel import random_pixel_perturbation
        from tools.os import rm_if_exist
        # craft AEs
        dir_sc = f'{exp.data_dir}/val2017'
        dir_dst = f'{exp.data_dir}_rand/val2017'
        rm_if_exist(dir_dst)
        os.makedirs(dir_dst, exist_ok=False)
        img_names = [f for f in os.listdir(dir_sc) if f.endswith(('.jpg', '.png'))]
        model.eval().cuda()
        for name in tqdm(img_names, desc="Generating Adversarial Examples"):
            img_path = os.path.join(dir_sc, name)
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).cuda()  # shape: [1, C, H, W]
            ori_shape = (img_tensor.shape[-2], img_tensor.shape[-1])
            img_tensor = transforms.Resize((640, 640))(img_tensor)
            adv_tensor = random_pixel_perturbation(img_tensor, epslon=5)  # shape: [1, C, H, W]
            adv_tensor = adv_tensor.squeeze(0).detach().cpu().clamp(torch.min(img_tensor).cpu(), torch.max(img_tensor).cpu())
            adv_tensor = transforms.Resize(ori_shape)(adv_tensor)
            adv_img = transforms.ToPILImage()(adv_tensor)
            adv_img.save(os.path.join(dir_dst, name))
        exp.data_dir = f'{exp.data_dir}_rand/'

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True


    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
