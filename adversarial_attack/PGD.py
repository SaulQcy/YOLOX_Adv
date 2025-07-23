import torch

from .__init__ import *



def pgd_attack(img, model, opt):
    model.eval()
    device = next(model.parameters()).device
    alpha = 0.01
    k = opt.N
    num_iter = opt.num_iter
    t_min = torch.min(img)
    t_max = torch.max(img)
    decay = 1.0
    eps = opt.max_epsilon / 255.0 if t_max <= 1 else opt.max_epsilon

    x_nat = img.clone().detach().to(device)
    x_adv = x_nat + torch.empty_like(x_nat).uniform_(-eps, eps)
    # x_adv = torch.clamp(x_adv, -1., 1.)
    x_adv = clip_by_tensor(x_adv, t_min, t_max)
    with torch.no_grad():
        y = model(x_nat)
    momentum = torch.zeros_like(x_adv, device=x_nat.device)

    for _ in range(num_iter):
        noise = 0.
        total = 0
        for _ in range(k):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            outputs = model(x_adv)
            if outputs is None:
                continue
            loss = torch.tensor([0.]).cuda()
            loss += F.mse_loss(outputs, y)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                noise += grad
                total += 1
        if total > 0:
            noise /= total
        else:
            continue
        if 'Linf' in opt.attack:
            x_adv = x_adv + alpha * torch.sign(noise)  # Step in sign(gradient) direction
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)  # Clip to Linf ball
        elif 'L2' in opt.attack:
            delta = (alpha * torch.sign(noise)).renorm(p=2, dim=0, maxnorm=eps)
            x_adv = torch.clamp(x_adv + delta, -1, 1).detach()
        elif 'L1' in opt.attack:
            delta = (alpha * torch.sign(noise)).renorm(p=1, dim=0, maxnorm=eps)
            x_adv = torch.clamp(x_adv + delta, -1, 1).detach()
        elif 'MI' in opt.attack:
            noise_norm = noise / (torch.mean(torch.abs(noise), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = decay * momentum + noise_norm
            x_adv = x_adv.detach() + alpha * torch.sign(momentum)
            x_adv = torch.clamp(x_adv, x_nat - eps, x_nat + eps).clamp(-1, 1).detach().requires_grad_(True)
        else:
            raise NotImplementedError(opt.attack)

        # x_adv = torch.clamp(x_adv, -1., 1.)
        clip_by_tensor(x_adv, t_min, t_max)
    return x_adv.detach()
