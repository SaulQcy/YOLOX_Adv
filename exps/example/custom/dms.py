#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        self.act = "relu"

        self.data_num_workers = 8

        self.num_classes = 80

        self.data_dir = "/home/saul/proj/YOLOX/datasets/COCO2017"
        self.train_ann = "/home/saul/proj/YOLOX/datasets/COCO2017/annotations/instances_train2017.json"
        self.val_ann = "/home/saul/proj/YOLOX/datasets/COCO2017/annotations/instances_val2017.json"
        self.test_ann = None

        # self.warmup_epochs = 5
        # self.max_epoch = 50

        self.warmup_epochs = 1  # for free adv traiing
        self.max_epoch = 10  # for free adv traiing
        
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.01 / 64.0
        self.no_aug_epochs = 15

        self.print_interval = 10
        self.eval_interval = 10

        self.save_history_ckpt = False

        self.test_size = (640, 640)

        # self.input_size = (640, 640)
        # self.multiscale_range = 0


    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOFPN_DMS, YOLOXHead
            backbone = YOLOFPN_DMS()
            head = YOLOXHead(self.num_classes, self.width, in_channels=[128, 128, 128], act="relu")
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model
