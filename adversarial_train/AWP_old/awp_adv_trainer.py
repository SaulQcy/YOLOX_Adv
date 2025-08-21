import time

import torch
from yolox.core.trainer import Trainer
from yolox.exp.yolox_base import Exp
import random
from tools.os import manual_seed
from tqdm import tqdm
from adversarial_train.AWP.utils_awp import TradesAWP

class Args:
    pass

class Awp_Adv_Trainer(Trainer):
    def __init__(self, exp: Exp, args):
        manual_seed(42)
        super().__init__(exp, args)

        self.opt = Args()

        self.opt.__dict__ = {
            'attack': 'PGD_Linf',
            'N': 5,
            'sigma': 16.0,
            'momentum': 1.0,
            'num_iter': 4,
            'max_epsilon': 8.,
            'rho': 0.5,
            'bs': 8,
            'max_len': 2000,
            'ar': 0.3
        }
        # init proxy model
        self.proxy = None
        self.proxy_optim = None
        self.awp_gamma = None
        self.beta = None
        self.awp_adversary = TradesAWP(model=self.model, 
                                       proxy=self.proxy, 
                                       proxy_optim=self.proxy_optim, 
                                       gamma=self.awp_gamma)
    
    def train_one_iter(self):
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        # PGD adversarial training parameters
        eps = self.opt.max_epsilon / 255.     # e.g., 8/255
        alpha = 0.01
        num_steps = self.opt.num_iter

        x_nat = inps.detach()
        if random.random() < self.opt.ar:
            delta = torch.zeros_like(x_nat).uniform_(-eps, eps).to(x_nat.device)
            delta.requires_grad = True
            # gen. adv. sample
            self.model.eval()
            for _ in range(num_steps):
                adv = (x_nat + delta).clamp(0.0, 1.0).detach()
                adv = torch.max(torch.min(adv, x_nat + eps), x_nat - eps)
                adv.requires_grad_()


                outputs = self.model(adv, targets)
                loss = outputs["total_loss"]

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                grad = adv.grad.detach()
                delta.data = delta + alpha * torch.sign(grad)
                delta.data = torch.clamp(delta, -eps, eps)
                delta.grad = None

            # Final update using adversarial examples
            x_adv = (x_nat + delta).clamp(0.0, 1.0).detach()
            self.model.train()
            awp = self.awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=x_nat,
                                         beta=self.beta)
            self.awp_adversary.perturb(awp)
        else:
            x_adv = x_nat.detach()
        outputs = self.model(x_adv, targets)
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
