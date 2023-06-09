import torch
from torch import nn
import torch.nn.functional as F


def gradient(img):
    laplace_filter = torch.nn.Conv2d(1, 1, (3, 3), stride=2, padding=1, bias=False)
    laplace_filter.weight.data = torch.Tensor([[[[0, -1, 0],
                                                 [-1, 4, -1],
                                                 [0, -1, 0]]]]).cuda()
    return laplace_filter(img)


class ContentLoss(nn.Module):

    def __init__(self, _lambda):
        super(ContentLoss, self).__init__()
        self._lambda = _lambda

    def forward(self, img_vi, img_ir, img_fu):
        gradient_loss = (gradient(img_fu) - gradient(img_vi)).square().mean()
        pixel_loss = (img_fu - img_ir).square().mean()
        return self._lambda * gradient_loss + pixel_loss


class ContentLoss1(nn.Module):
    """
    fusion-gan 这篇论文中为generator定义的损失函数，我个人的pytorch实现。
    目的是使生成的图像的像素尽可能接近红外图像，同时使图像的梯度接近可见光图像。
    梯度使用的是laplace算符实现的。
    """

    def __init__(self, _lambda):
        super(ContentLoss1, self).__init__()
        self._lambda = _lambda

    def forward(self, img_vi, img_ir, img_fu):
        gradient_loss = (self._gradient(img_fu) - self._gradient(img_vi)).square().mean()
        pixel_loss = (img_fu - img_ir).square().mean()
        return self._lambda * gradient_loss + pixel_loss

    def _gradient(self, img):
        laplace_filter = torch.tensor([[[[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]]]])

        out = F.conv2d(img, laplace_filter, stride=2, padding=1)
        return out

class PixelLoss(nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()

    def forward(self, gen_img, origin_img):
        pixel_loss = (origin_img - gen_img).square().mean()
        pixel_loss.requires_grad_(True)
        return pixel_loss

