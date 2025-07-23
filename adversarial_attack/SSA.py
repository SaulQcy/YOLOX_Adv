from .__init__ import  *

def SSA_attack(img, obj, opt):
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

    images_min = clip_by_tensor(img - opt.max_epsilon / 255.0, -1., 1.0)
    images_max = clip_by_tensor(img + opt.max_epsilon / 255.0, -1., 1.0)

    for i in range(num_iter):
        noise = 0.
        for n in range(N):
            gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask).clone().detach().requires_grad_(True)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            # output_v3 = model(DI(x_idct))

            output_v3 = model(x_idct)
            output_v3 = postprocess(
                output_v3, obj.num_classes, obj.confthre,
                obj.nmsthre, class_agnostic=True
            )[0]
            if output_v3 is None:
                break
            l = min(output_v3.shape[0], gt.shape[0])
            loss = 0.
            for i in range(l):
                loss1 = F.mse_loss(output_v3[i], gt[i])
                loss2 = F.mse_loss(
                    output_v3[i][4] * output_v3[i][5],
                    gt[i][4] * gt[i][5],
                )
                loss = loss  + loss1 + loss2 * 1e2
            # action-label
            # loss.backward()
            grad = torch.autograd.grad(loss, x_idct, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                noise += grad
        noise = noise / N
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, images_min.to(x), images_max.to(x))
    return x.detach()


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real

