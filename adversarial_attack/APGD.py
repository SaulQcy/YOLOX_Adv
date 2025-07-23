from .__init__ import *

def apgd_attack(x_nat, obj, opt):
    model = obj.model
    model.eval()
    device = x_nat.device
    eps = opt.max_epsilon / 255.0
    alpha = opt.alpha if hasattr(opt, 'alpha') else eps / 10.
    n_iter = opt.n_iter if hasattr(opt, 'n_iter') else 10
    n_restarts = opt.n_restarts if hasattr(opt, 'n_restarts') else 1
    tau = 2 / n_iter  # APGD uses adaptive oscillation threshold
    best_adv = x_nat.clone()
    best_loss = torch.full((x_nat.shape[0],), float('-inf'), device=device)

    with torch.no_grad():
        y = model(x_nat)
    y = postprocess(y, obj.num_classes, obj.confthre, obj.nmsthre, class_agnostic=True)[0]

    for _ in range(n_restarts):
        x_adv = x_nat + torch.empty_like(x_nat).uniform_(-eps, eps)
        x_adv = x_adv.clamp(-1., 1.).detach()
        step_size = alpha

        loss_old = None
        for i in range(n_iter):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            out = model(x_adv)
            out = postprocess(out, obj.num_classes, obj.confthre, obj.nmsthre, class_agnostic=True)[0]
            if out is None:
                break
            l = min(out.shape[0], y.shape[0])
            loss = 0.
            for j in range(l):
                loss1 = F.mse_loss(out[j], y[j])
                loss2 = F.mse_loss(out[j][4] * out[j][5], y[j][4] * y[j][5])
                loss += (loss1 + loss2 * 1e3)

            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + step_size * torch.sign(grad)
            x_adv = torch.clamp(x_adv, x_nat - eps, x_nat + eps).clamp(-1., 1.).detach()

            # Oscillation check for step adaptation
            if loss_old is not None:
                if (loss - loss_old).abs() / (loss_old.abs() + 1e-12) < tau:
                    step_size *= 0.9
            loss_old = loss

            # Track best adversarial
            if loss > best_loss.mean():
                best_adv = x_adv.clone().detach()
                best_loss[:] = loss.detach()
    return best_adv.detach()
