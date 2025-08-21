import time
import torch
from yolox.core.trainer import Trainer
from yolox.exp.yolox_base import Exp
import random
from tools.os import manual_seed
from tools.dct import dct_2d, idct_2d
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision



class Args:
    pass

OPT = Args()
OPT.__dict__ = {
    'M': 10,
    'max_epsilon': 5.,
}

# class Adv_Trainer(Trainer):
#     def __init__(self, exp: Exp, args):
#         manual_seed(42)
#         super().__init__(exp, args)
#         self.adv_rate = args.adv_rate  # e.g., 0.1
#         self.adv_mode = args.adv_mode  # e.g., 1
#         print(self.adv_mode, self.adv_rate)
#         self.opt = OPT
#         if self.adv_mode == 1:  # PGD_Linf
#             from adversarial_attack.PGD import pgd_attack
#             self.generate_AEs = pgd_attack
#         else:
#             raise NotImplementedError
    
#     def train_one_iter(self):
#         iter_start_time = time.time()
#         inps, targets = self.prefetcher.next()
#         inps = inps.to(self.data_type)
#         targets = targets.to(self.data_type)
#         targets.requires_grad = False
#         inps, targets = self.exp.preprocess(inps, targets, self.input_size)
#         data_end_time = time.time()

#         # torchvision.utils.save_image(inps.cpu().detach(), 'AA.png', nrow=4)


#         # adversarial training
#         if self.epoch > 5 and self.epoch % 2 == 0 and random.random() < self.adv_rate:
#             inps = self.generate_AEs(inps, self.model, self.opt).clone().detach()
#             self.model.train()
#         with torch.cuda.amp.autocast(enabled=self.amp_training):
#             outputs = self.model(inps, targets)

#         loss = outputs["total_loss"]

#         self.optimizer.zero_grad()
#         self.scaler.scale(loss).backward()
#         self.scaler.step(self.optimizer)
#         self.scaler.update()

#         if self.use_model_ema:
#             self.ema_model.update(self.model)

#         lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
#         for param_group in self.optimizer.param_groups:
#             param_group["lr"] = lr

#         iter_end_time = time.time()
#         self.meter.update(
#             iter_time=iter_end_time - iter_start_time,
#             data_time=data_end_time - iter_start_time,
#             lr=lr,
#             **outputs,
#         )

class Free_Adv_Trainer(Trainer):
    def __init__(self, exp: Exp, args):
        manual_seed(42)
        super().__init__(exp, args)
        self.opt = OPT
    
    def train_one_iter(self):
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()
        x_nat = inps.detach()
        glob_noise = torch.zeros_like(x_nat).to(x_nat)
        epslon = self.opt.max_epsilon if torch.max(x_nat) > 1 else self.opt.max_epsilon / 255.
        t_min = torch.min(x_nat)
        t_max = torch.max(x_nat)
        alpha = epslon / self.opt.M

        for _ in range(self.opt.M):
            # print(glob_noise[0, 0, 0:3, 0:3])
            noise_batch = glob_noise.clone().detach().to(x_nat.device)
            noise_batch.requires_grad = True
            noise_batch.retain_grad()
            x_adv = x_nat + noise_batch
            x_adv = x_adv.clamp(t_min, t_max)
            outputs = self.model(x_adv, targets)
            loss = outputs["total_loss"]

            # update model param.
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # update pert.
            grad = noise_batch.grad
            # pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            # pert = 0.05 * torch.sign(grad)
            pert = alpha * torch.sign(grad)
            glob_noise += pert.data
            glob_noise.clamp_(-epslon, epslon)

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
class Baseline_Free_Adv_Trainer(Trainer):
    def __init__(self, exp: Exp, args):
        manual_seed(42)
        super().__init__(exp, args)
        self.opt = OPT
    
    def train_one_iter(self):
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()
        x_nat = inps.detach()
        t_min = torch.min(x_nat)
        t_max = torch.max(x_nat)
        alpha = epslon / self.opt.M
        for _ in range(self.opt.M):
            # print(i)
            # print(glob_noise[0, 0, 0:3, 0:3])
            x_adv = x_nat
            x_adv = x_adv.clamp(t_min, t_max)
            outputs = self.model(x_adv, targets)
            loss = outputs["total_loss"]

            # update model param.
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

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
class Mix_Free_Adv_Trainer(Trainer):
    def __init__(self, exp: Exp, args):
        manual_seed(42)
        super().__init__(exp, args)
        self.opt = OPT
    
    def train_one_iter(self):
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()
        x_nat = inps.detach()
        # glob_noise = torch.zeros_like(x_nat).to(x_nat)

        noise_init = random.random()
        epslon = self.opt.max_epsilon if torch.max(x_nat) > 1 else self.opt.max_epsilon / 255.
        alpha = epslon / self.opt.M
        t_min = torch.min(x_nat)
        t_max = torch.max(x_nat)
        if 0 <= noise_init < 0.5:
            # 40% zero init
            glob_noise = torch.zeros_like(x_nat).to(x_nat)
        # elif 0.4 <= noise_init < 0.6:
        #     # 20% Gaussian init (clipped to [-epslon, epslon])
        #     glob_noise = torch.randn_like(x_nat) * (epslon / 3)  
        #     glob_noise = torch.clamp(glob_noise, -epslon, epslon)
        # elif 0.6 <= noise_init < 0.8:
        #     # 20% Uniform init in [-epslon, epslon]
        #     glob_noise = torch.empty_like(x_nat).uniform_(-epslon, epslon)
        elif 0.5 <= noise_init < 1.:
            # 20% random pixel perturbation (already bounded by function)
            from adversarial_attack.random_pixel import random_pixel_perturbation
            tmp = random_pixel_perturbation(x_nat.clone(), 5, 0.9)
            # torchvision.utils.save_image(tmp / 255., "test.png")
            glob_noise = x_nat - tmp
            # print(torch.max(glob_noise))
            glob_noise = torch.clamp(glob_noise, -epslon, epslon)  # Ensure bounding
        else:
            raise ValueError(noise_init)
        for _ in range(self.opt.M):
            # print(glob_noise[0, 0, 0:3, 0:3])
            noise_batch = glob_noise.clone().detach().to(x_nat.device)
            noise_batch.requires_grad = True
            noise_batch.retain_grad()
            x_adv = x_nat + noise_batch
            x_adv = x_adv.clamp(t_min, t_max)
            outputs = self.model(x_adv, targets)
            loss = outputs["total_loss"]

            # update model param.
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # update pert.
            grad = noise_batch.grad
            # pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            pert = alpha * torch.sign(grad)
            glob_noise += pert.data
            glob_noise.clamp_(-epslon, epslon)

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
# class Frequency_Free_Adv_Trainer(Trainer):
#     def __init__(self, exp: Exp, args):
#         manual_seed(42)
#         super().__init__(exp, args)
#         self.opt = OPT
#         self.M = 5
#         self.rho = 0.5
    
#     def train_one_iter(self):
#         iter_start_time = time.time()
#         inps, targets = self.prefetcher.next()
#         inps = inps.to(self.data_type)
#         targets = targets.to(self.data_type)
#         targets.requires_grad = False
#         inps, targets = self.exp.preprocess(inps, targets, self.input_size)
#         data_end_time = time.time()
#         x_nat = inps.detach()
#         glob_noise = torch.zeros_like(x_nat).to(x_nat)
#         epslon = self.opt.max_epsilon if torch.max(x_nat) > 1 else self.opt.max_epsilon / 255.
#         for _ in range(self.M):
#             # print(glob_noise[0, 0, 0:3, 0:3])
#             noise_batch = glob_noise.clone().detach().to(x_nat.device)
#             noise_batch.requires_grad = True
#             noise_batch.retain_grad()
#             x_adv = x_nat + noise_batch

#             if random.random() < 0.5:
#                 # SSA attack
#                 x_dct = dct_2d(x_adv)
#                 mask = (torch.rand_like(x_nat) * 2 * self.rho + 1 - self.rho).cuda()
#                 x_adv = idct_2d(x_dct * mask)
            
#             x_adv = x_adv.clamp(0., 255.)
#             outputs = self.model(x_adv, targets)
#             loss = outputs["total_loss"]

#             # update model param.
#             self.optimizer.zero_grad()
#             self.scaler.scale(loss).backward()
#             self.scaler.unscale_(self.optimizer)

#             # update pert.
#             grad = noise_batch.grad
#             # pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
#             pert = 0.05 * torch.sign(grad)
#             glob_noise += pert.data
#             glob_noise.clamp_(-epslon, epslon)

#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#             if self.use_model_ema:
#                 self.ema_model.update(self.model)

#             lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
#             for param_group in self.optimizer.param_groups:
#                 param_group["lr"] = lr

#         iter_end_time = time.time()
#         self.meter.update(
#             iter_time=iter_end_time - iter_start_time,
#             data_time=data_end_time - iter_start_time,
#             lr=lr,
#             **outputs,
#         )
# class Linf_Mix_Free_Adv_Trainer(Trainer):
#     def __init__(self, exp: Exp, args):
#         manual_seed(42)
#         super().__init__(exp, args)
#         self.opt = OPT
#         self.M = 5
    
#     def train_one_iter(self):
#         iter_start_time = time.time()
#         inps, targets = self.prefetcher.next()
#         inps = inps.to(self.data_type)
#         targets = targets.to(self.data_type)
#         targets.requires_grad = False
#         inps, targets = self.exp.preprocess(inps, targets, self.input_size)
#         data_end_time = time.time()
#         x_nat = inps.detach()
#         # glob_noise = torch.zeros_like(x_nat).to(x_nat)

#         noise_init = random.random()
#         epslon = self.opt.max_epsilon if torch.max(x_nat) > 1 else self.opt.max_epsilon / 255.

#         if 0 <= noise_init < 0.5:
#             # 40% zero init
#             glob_noise = torch.zeros_like(x_nat).to(x_nat)
#         # elif 0.4 <= noise_init < 0.6:
#         #     # 20% Gaussian init (clipped to [-epslon, epslon])
#         #     glob_noise = torch.randn_like(x_nat) * (epslon / 3)  
#         #     glob_noise = torch.clamp(glob_noise, -epslon, epslon)
#         # elif 0.6 <= noise_init < 0.8:
#         #     # 20% Uniform init in [-epslon, epslon]
#         #     glob_noise = torch.empty_like(x_nat).uniform_(-epslon, epslon)
#         elif 0.5 <= noise_init < 1.:
#             # 20% random pixel perturbation (already bounded by function)
#             from adversarial_attack.random_pixel import random_pixel_perturbation
#             tmp = random_pixel_perturbation(x_nat.clone(), 5, 0.9)
#             glob_noise = x_nat - tmp
#             glob_noise = torch.clamp(glob_noise, -epslon, epslon)  # Ensure bounding
#         else:
#             raise ValueError(noise_init)
#         # print(noise_init)
        
#         for _ in range(self.M):
#             # print(glob_noise[0, 0, 0:3, 0:3])
#             noise_batch = glob_noise.clone().detach().to(x_nat.device)
#             noise_batch.requires_grad = True
#             noise_batch.retain_grad()
#             x_adv = x_nat + noise_batch
#             x_adv = torch.max(torch.min(x_adv, x_nat + epslon), x_nat - epslon)
#             # x_adv = x_adv.clamp(0., 255.)
#             outputs = self.model(x_adv, targets)
#             loss = outputs["total_loss"]

#             # update model param.
#             self.optimizer.zero_grad()
#             self.scaler.scale(loss).backward()
#             self.scaler.unscale_(self.optimizer)

#             # update pert.
#             grad = noise_batch.grad
#             # pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
#             pert = 0.05 * torch.sign(grad)
#             glob_noise += pert.data
#             glob_noise.clamp_(-epslon, epslon)

#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#             if self.use_model_ema:
#                 self.ema_model.update(self.model)

#             lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
#             for param_group in self.optimizer.param_groups:
#                 param_group["lr"] = lr

#         iter_end_time = time.time()
#         self.meter.update(
#             iter_time=iter_end_time - iter_start_time,
#             data_time=data_end_time - iter_start_time,
#             lr=lr,
#             **outputs,
#         )
class Improved_Mix_Free_Adv_Trainer(Trainer):
    def __init__(self, exp: Exp, args):
        manual_seed(42)
        super().__init__(exp, args)
        self.opt = OPT
        assert self.opt.M == self.exp.print_interval
        self.global_noise = None
    
    def train_one_iter(self):
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()
        x_nat = inps.detach()
        # glob_noise = torch.zeros_like(x_nat).to(x_nat)

        epslon = self.opt.max_epsilon if torch.max(x_nat) > 1 else self.opt.max_epsilon / 255.
        alpha = epslon / self.opt.M * 2
        t_min = torch.min(x_nat)
        t_max = torch.max(x_nat)

        # global noise init:
        if self.global_noise is None or (isinstance(self.global_noise, torch.Tensor) and self.global_noise.shape != x_nat.shape):
            self.global_noise = init_noise(x_nat, epslon)
        
        # adv train
        # print(self.global_noise[0, 0, 0:3, 0:3])
        noise_batch = self.global_noise.clone().detach().to(x_nat.device)
        noise_batch.requires_grad = True
        noise_batch.retain_grad()
        x_adv = x_nat + noise_batch
        x_adv = x_adv.clamp(t_min, t_max)
        outputs = self.model(x_adv, targets)
        loss = outputs["total_loss"]

        # update model param.
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # update pert.
        grad = noise_batch.grad
        # pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
        pert = alpha * torch.sign(grad)
        self.global_noise += pert.data
        self.global_noise.clamp_(-epslon, epslon)

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

from adversarial_train.AWP.awp import AWP
class AWP_Free_Adv_Trainer(Trainer):
    def __init__(self, exp: Exp, args):
        manual_seed(42)
        super().__init__(exp, args)
        self.opt = OPT
        assert self.opt.M == self.exp.print_interval
        self.global_noise = None
        self.awp = None
        self.awp_epoch = 10
    
    def train_one_iter(self):
        if self.awp is None and self.epoch > self.awp_epoch:
            self.awp = AWP(self.model, self.optimizer, adv_lr=5e-3, adv_eps=0.001)
            print("==> awp is built.")
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()
        x_nat = inps.detach()
        # glob_noise = torch.zeros_like(x_nat).to(x_nat)

        epslon = self.opt.max_epsilon if torch.max(x_nat) > 1 else self.opt.max_epsilon / 255.
        alpha = epslon / self.opt.M * 2
        t_min = torch.min(x_nat)
        t_max = torch.max(x_nat)

        # global noise init:
        if self.global_noise is None or (isinstance(self.global_noise, torch.Tensor) and self.global_noise.shape != x_nat.shape):
            self.global_noise = init_noise(x_nat, epslon)
        
        # adv train
        # print(self.global_noise[0, 0, 0:3, 0:3])
        noise_batch = self.global_noise.clone().detach().to(x_nat.device)
        noise_batch.requires_grad = True
        noise_batch.retain_grad()
        x_adv = x_nat + noise_batch
        x_adv = x_adv.clamp(t_min, t_max)
        outputs = self.model(x_adv, targets)
        loss = outputs["total_loss"]

        # update model param.
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward(retain_graph=True)
        self.scaler.unscale_(self.optimizer)

        if self.epoch > self.awp_epoch and self.awp is not None:
            loss = self.awp.attack_backward(inps=x_adv, targets=targets)
            if loss is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            self.awp._restore()     

        # update pert.
        grad = noise_batch.grad
        pert = alpha * torch.sign(grad)
        self.global_noise += pert.data
        self.global_noise.clamp_(-epslon, epslon)

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
  


def init_noise(x_nat, epslon):
    noise_init = random.random()
    if 0 <= noise_init < 0.5:
        # 40% zero init
        tmp = torch.zeros_like(x_nat).to(x_nat)
    elif 0.5 <= noise_init < 1.:
        # 20% random pixel perturbation (already bounded by function)
        from adversarial_attack.random_pixel import random_pixel_perturbation
        tmp = random_pixel_perturbation(x_nat.clone(), 5, 0.9)
        # torchvision.utils.save_image(tmp / 255., "test.png")
        tmp = x_nat - tmp
        # print(torch.max(glob_noise))
        tmp = torch.clamp(tmp, -epslon, epslon)  # Ensure bounding
        # print(tmp)
    else:
        raise ValueError(noise_init)
    return tmp


