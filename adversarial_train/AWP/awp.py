import torch

class AWP:
    """
    Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1., adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inps, targets):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        outputs = self.model(inps, targets)
        awp_loss = outputs["total_loss"]
        # self.optimizer.zero_grad()
        return awp_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

"""
    使用AWP的训练过程
"""
# # 初始化AWP
# awp = AWP(model, loss_fn, optimizer, adv_lr=awp_lr, adv_eps=awp_eps)

# for step, batch in enumerate(train_loader):
#     inputs, labels = batch
    
#     # 将模型的参数梯度初始化为0
#     optimizer.zero_grad()
    
#     # forward + backward + optimize
#     predicts = model(inputs)          # 前向传播计算预测值
#     loss = loss_fn(predicts, labels)  # 计算当前损失
#     loss.backward()       # 反向传播计算梯度
#     # 指定从第几个epoch开启awp，一般先让模型学习到一定程度之后
#     if awp_start >= epoch:
#         loss = awp.attack_backward(inputs, labels)
#         loss.backward()
#         awp._restore()                    # 恢复到awp之前的model
#     optimizer.step()                  # 更新所有参数 