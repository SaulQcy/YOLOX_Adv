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

from adversarial_train.AWP.utils_awp import TradesAWP


class Args:
    pass

OPT = Args()
OPT.__dict__ = {
    'M': 5,
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



# class AWP_Adv_Trainer(Trainer):
#     def __init__(self, exp: Exp, args):
#         manual_seed(42)
#         super().__init__(exp, args)
#         self.adv_rate = args.adv_rate  # e.g., 0.1
#         self.adv_mode = args.adv_mode  # e.g., 1
#         print(self.adv_mode, self.adv_rate)
#         self.opt = OPT
        
#     def before_train(self):
#         super().before_train()
#         # init proxy model
#         self.proxy = self.exp.get_model()
#         if self.exp.warmup_epochs > 0:
#             lr = self.exp.warmup_lr
#         else:
#             lr = self.exp.basic_lr_per_img * 16
#         pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

#         for k, v in self.proxy.named_modules():
#             if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
#                 pg2.append(v.bias)  # biases
#             if isinstance(v, nn.BatchNorm2d) or "bn" in k:
#                 pg0.append(v.weight)  # no decay
#             elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
#                 pg1.append(v.weight)  # apply decay
#         print(f'pg0 length: {len(pg0)}')
#         print(f'pg1 length: {len(pg1)}')
#         print(f'pg2 length: {len(pg2)}')

#         optimizer = torch.optim.SGD(
#             pg0, lr=lr, momentum=self.exp.momentum, nesterov=True
#         )
#         optimizer.add_param_group(
#             {"params": pg1, "weight_decay": self.exp.weight_decay}
#         )  # add pg1 with weight_decay
#         optimizer.add_param_group({"params": pg2})
#         self.proxy_optim = optimizer

#         self.awp_gamma = 0.005
#         self.beta = 6.
#         self.awp_adversary = TradesAWP(model=self.model, 
#                                        proxy=self.proxy, 
#                                        proxy_optim=self.proxy_optim, 
#                                        gamma=self.awp_gamma)
        
    
#     def train_one_iter(self):
#         iter_start_time = time.time()
#         inps, targets = self.prefetcher.next()
#         inps = inps.to(self.data_type)
#         targets = targets.to(self.data_type)
#         targets.requires_grad = False
#         inps, targets = self.exp.preprocess(inps, targets, self.input_size)
#         data_end_time = time.time()

#         # PGD adversarial training parameters
#         alpha = 0.01
#         num_steps = self.opt.num_iter

#         x_nat = inps.detach()
#         torchvision.utils.save_image(x_nat.cpu().detach(), 'AA.png', nrow=4)

#         t_min = torch.min(x_nat)
#         t_max = torch.max(x_nat)
#         eps = self.opt.max_epsilon / 255. if t_max <=1 else self.opt.max_epsilon

#         if random.random() < self.opt.ar:
#             # gen. adv. sample
#             x_adv = x_nat.detach() + 0.001 * torch.zeros_like(x_nat).uniform_(-eps, eps).to(x_nat.device)
#             self.model.eval()
#             for _ in range(num_steps):
#                 x_adv = x_adv.clone().detach().requires_grad_(True)
#                 loss = self.model(x_adv, targets).mean()
#                 grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
#                 noise = alpha * torch.sign(grad.detach())
#                 x_adv = x_adv.detach() + noise
#                 x_adv = torch.min(torch.max(x_adv, x_nat - eps), x_nat + eps)
#                 x_adv = torch.clamp(x_adv, t_min, t_max)

#             # Final update using adversarial examples
#             self.model.train()
#             x_adv = x_adv.clone().detach().requires_grad_(True)
#             awp = self.awp_adversary.calc_awp(inputs_adv=x_adv,
#                                          inputs_clean=x_nat,
#                                          target=targets,
#                                          beta=self.beta)
#             self.awp_adversary.perturb(awp)
#         else:
#             x_adv = x_nat.detach()
#         outputs = self.model(x_adv, targets)
#         loss = outputs["total_loss"]

#         self.optimizer.zero_grad()
#         self.scaler.scale(loss).backward()
#         self.scaler.unscale_(self.optimizer)
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

#     def resume_train(self, model):
#         model = super().resume_train(model)
#         if self.args.resume:
#             if self.args.ckpt is None:
#                 import os
#                 ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
#             else:
#                 ckpt_file = self.args.ckpt

#             ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
#             self.awp_adversary = ckpt['awp_adversary']
#         return model

#     def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
#         if self.rank == 0:
#             save_model = self.ema_model.ema if self.use_model_ema else self.model
#             ckpt_state = {
#                 "start_epoch": self.epoch + 1,
#                 "model": save_model.state_dict(),
#                 "optimizer": self.optimizer.state_dict(),
#                 "best_ap": self.best_ap,
#                 "curr_ap": ap,
#                 # for AWP adversarial training
#                 "awp_adversary": self.awp_adversary,
#             }
#             from yolox.utils.checkpoint import save_checkpoint
#             save_checkpoint(
#                 ckpt_state,
#                 update_best_ckpt,
#                 self.file_name,
#                 ckpt_name,
#             )

#             # if self.args.logger == "wandb":
#             #     self.wandb_logger.save_checkpoint(
#             #         self.file_name,
#             #         ckpt_name,
#             #         update_best_ckpt,
#             #         metadata={
#             #             "epoch": self.epoch + 1,
#             #             "optimizer": self.optimizer.state_dict(),
#             #             "best_ap": self.best_ap,
#             #             "curr_ap": ap
#             #         }
#             #     )


# class Free_AWP_Adv_Trainer(Trainer):
#     def __init__(self, exp: Exp, args):
#         manual_seed(42)
#         super().__init__(exp, args)
#         self.opt = OPT
#         self.M = 8
    
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
#         t_max = torch.max(x_nat)
#         t_min = torch.min(x_nat)
#         epslon = self.opt.max_epsilon if t_max > 1 else self.opt.max_epsilon / 255.
#         for _ in range(self.M):
#             noise_batch = glob_noise.clone().detach().to(x_nat.device)
#             noise_batch.requires_grad = True
#             noise_batch.retain_grad()
#             x_adv = x_nat + noise_batch
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
#             pert = 0.01 * torch.sign(grad)
#             glob_noise[0:inps.size(0)] += pert.data
#             glob_noise.clamp_(-epslon, epslon)

#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#             if self.use_model_ema:
#                 self.ema_model.update(self.model)

#         self.model.train()
#         x_adv = x_nat + noise_batch
#         x_adv = x_adv.clamp(t_min, t_max)
#         x_adv = x_adv.clone().detach().requires_grad_(True)
#         awp = self.awp_adversary.calc_awp(inputs_adv=x_adv,
#                                         inputs_clean=x_nat,
#                                         target=targets,
#                                         beta=self.beta)
#         self.awp_adversary.perturb(awp)
#         outputs = self.model(x_adv, targets)
#         loss = outputs["total_loss"]

#         self.optimizer.zero_grad()
#         self.scaler.scale(loss).backward()
#         self.scaler.unscale_(self.optimizer)
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

#     def before_train(self):
#         super().before_train()
#         # init proxy model
#         self.proxy = self.exp.get_model()
#         if self.exp.warmup_epochs > 0:
#             lr = self.exp.warmup_lr
#         else:
#             lr = self.exp.basic_lr_per_img * 16
#         pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

#         for k, v in self.proxy.named_modules():
#             if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
#                 pg2.append(v.bias)  # biases
#             if isinstance(v, nn.BatchNorm2d) or "bn" in k:
#                 pg0.append(v.weight)  # no decay
#             elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
#                 pg1.append(v.weight)  # apply decay
#         print(f'pg0 length: {len(pg0)}')
#         print(f'pg1 length: {len(pg1)}')
#         print(f'pg2 length: {len(pg2)}')

#         optimizer = torch.optim.SGD(
#             pg0, lr=lr, momentum=self.exp.momentum, nesterov=True
#         )
#         optimizer.add_param_group(
#             {"params": pg1, "weight_decay": self.exp.weight_decay}
#         )  # add pg1 with weight_decay
#         optimizer.add_param_group({"params": pg2})
#         self.proxy_optim = optimizer

#         self.awp_gamma = 0.005
#         self.beta = 6.
#         self.awp_adversary = TradesAWP(model=self.model, 
#                                        proxy=self.proxy, 
#                                        proxy_optim=self.proxy_optim, 
#                                        gamma=self.awp_gamma)

#     def resume_train(self, model):
#         model = super().resume_train(model)
#         if self.args.resume:
#             if self.args.ckpt is None:
#                 import os
#                 ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
#             else:
#                 ckpt_file = self.args.ckpt

#             ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
#             self.awp_adversary = ckpt['awp_adversary']
#         return model

#     def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
#         if self.rank == 0:
#             save_model = self.ema_model.ema if self.use_model_ema else self.model
#             ckpt_state = {
#                 "start_epoch": self.epoch + 1,
#                 "model": save_model.state_dict(),
#                 "optimizer": self.optimizer.state_dict(),
#                 "best_ap": self.best_ap,
#                 "curr_ap": ap,
#                 # for AWP adversarial training
#                 "awp_adversary": self.awp_adversary,
#             }
#             from yolox.utils.checkpoint import save_checkpoint
#             save_checkpoint(
#                 ckpt_state,
#                 update_best_ckpt,
#                 self.file_name,
#                 ckpt_name,
#             )


