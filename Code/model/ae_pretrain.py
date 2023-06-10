import os
import sys

import torch
from h5py.h5d import namedtuple
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from Code.data.mydataset import MyDataset
from Code.model import ops
from Code.model.backbones import Discriminator_FusionGan, AE_YGC
from Code.model.losses import ContentLoss1, PixelLoss
from Code.utils.visual import accuracy, Accumulator, Animator, evaluate_accuracy


class AE_pretrainer:
    def __init__(self, AE_path=None, **kwargs):
        self.cfg = self.parse_args(**kwargs)

        # GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # 准备模型
        self.AE = AE_YGC()
        ops.init_weights(self.AE)

        # 加载checkpoint
        if AE_path is not None:
            self.AE.load_state_dict(torch.load(
                AE_path, map_location=lambda storage, loc: storage))
        self.AE = self.AE.to(self.device)

        # 定义损失函数
        #self.AE_Loss = ContentLoss1(5, self.device)
        self.AE_Loss = PixelLoss()

        # 定义优化器
        self.optimizer_AE = optim.Adam(self.AE.parameters(), lr=self.cfg.lr, betas=(self.cfg.b1, self.cfg.b2))
    def train_step(self, vi_imgs, epoch, backward=True):
        self.AE.train(backward)

        # 生成图片
        gen_imgs = self.AE(vi_imgs)

        with torch.set_grad_enabled(backward):
            # 定义损失
            g_loss = self.AE_Loss(gen_imgs, vi_imgs)

            # 训练
            self.optimizer_AE.zero_grad()
            g_loss.backward()
            self.optimizer_AE.step()

            # 计算精度

            # 保存生成的图片
            if (epoch + 1) % 50 == 0:
                save_image(vi_imgs.data[:25], "../../Data/Vi_imgs/%d.png" % (epoch + 1), nrow=5, normalize=True)
                save_image(gen_imgs.data[:25], "../../Data/Gen_imgs/%d.png" % (epoch + 1), nrow=5, normalize=True)

            return g_loss.item()

    @torch.enable_grad()
    def train_over(self, save_dir='../../checkpoint'):
        self.AE.train()

        # 准备模型储存路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 准备数据集
        train_trans = transforms.ToTensor()
        train_dataset = MyDataset(txt_path=r"../../Data/train.txt", transform=train_trans)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # 训练
        for epoch in range(self.cfg.epoch_num):
            for i, (vi_imgs, _) in enumerate(train_dataloader):
                vi_imgs = vi_imgs.to(self.device)
                g_loss = self.train_step(vi_imgs, epoch)
                print('Epoch: {} [{}/{}] G_Loss: {:.5f}'.format(
                    epoch + 1, i + 1, len(train_dataloader), g_loss))
                sys.stdout.flush()

            if (epoch + 1) % 50 == 0:
                # 保存模型
                AE_path = os.path.join(save_dir, 'YGC_AE_e%d.pth' % (epoch + 1))
                torch.save(self.AE.state_dict(), AE_path)

    def parse_args(self, **kwargs):
        # 参数设置
        cfg = {
            'epoch_num': 2000,
            'batch_size': 32,
            'num_workers': 6,
            # 与优化相关的参数
            'lr': 0.0002,
            'b1': 0.5,
            'b2': 0.999}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
