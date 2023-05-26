import torch
from torch import nn


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
        return self._lambda *gradient_loss + pixel_loss