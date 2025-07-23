import torch

from adversarial_attack import  *
from adversarial_attack.ANDA.utils import *

def get_theta(i, j):
    theta = torch.tensor([[[1, 0, i], [0, 1, j]]], dtype=torch.float)
    return theta

def get_thetas(n, min_r=-0.5, max_r=0.5):
    range_r = torch.linspace(min_r, max_r, n)
    thetas = []
    for i in range_r:
        for j in range_r:
            thetas.append(get_theta(i, j))
    thetas = torch.cat(thetas, dim=0)
    return thetas

def ANDA_attack(img, obj, opt):
    model = obj.model
    model.eval()
    with torch.no_grad():
        gt = model(img.cuda())
    gt = postprocess(
        gt, obj.num_classes, obj.confthre,
        obj.nmsthre, class_agnostic=True
    )[0]
    image_width = img.shape[-1]
    num_iter = 10
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = img.clone().cuda()
    rho = opt.rho
    N = opt.N
    sigma = opt.sigma

    n_ens = 25
    aug_max = 0.3
    min_x = x - eps
    max_x = x + eps
    thetas = get_thetas(int(math.sqrt(n_ens)), -aug_max, aug_max)
    n_ens = thetas.shape[0]
    x_adv = x.clone()
    anda = ANDA(data_shape=(1, 3, 512, 512), device=torch.device('cuda'))
    for i in range(num_iter):
        x_adv_batch = x_adv.repeat(n_ens, 1, 1, 1).clone().detach().requires_grad_(True)
        aug_x_adv_batch = translation(thetas, x_adv_batch)
        gt_s = gt.unsqueeze(0).repeat(n_ens, 1, 1)
        output = model(aug_x_adv_batch)
        output = postprocess(
            output, obj.num_classes, obj.confthre,
            obj.nmsthre, class_agnostic=True
        )
        output = torch.stack(output, dim=0)
        assert output.shape == gt_s.shape
        loss = 0.
        for i in range(output.shape[0]):
            p1 = output[i]
            p2 = gt_s[i]
            l = min(p1.shape[0], p2.shape[0])
            for i in range(l):
                loss += F.mse_loss(p1[i], p2[i])
        new_grad = torch.autograd.grad(loss, x_adv_batch, retain_graph=False, create_graph=False)[0]
        anda.collect_model(new_grad)
        sample_noise = anda.noise_mean
        with torch.no_grad():
            x_adv = x_adv + alpha * sample_noise.sign()
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
            x_adv = torch.max(torch.min(x_adv, max_x), min_x).detach()
    return x_adv.detach()


