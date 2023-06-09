import os
import sys

import torch
from h5py.h5d import namedtuple
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from Code.data.mydataset import MyDataset
from Code.model import ops
from Code.model.backbones import Discriminator_FusionGan, AE_YGC
from Code.model.losses import ContentLoss, ContentLoss1, PixelLoss


class Rebuilder:
    def __init__(self, AE_path=None, D_path=None, **kwargs):
        self.cfg = self.parse_args(**kwargs)

        # GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # 准备模型
        self.AE = AE_YGC()
        self.D = Discriminator_FusionGan()
        ops.init_weights(self.AE)
        ops.init_weights(self.D)

        # 加载checkpoint
        if AE_path is not None:
            self.AE.load_state_dict(torch.load(
                AE_path, map_location=lambda storage, loc: storage))
        if D_path is not None:
            self.D.load_state_dict(torch.load(
                D_path, map_location=lambda storage, loc: storage))
        self.AE = self.AE.to(self.device)
        self.D = self.D.to(self.device)

        # 定义损失函数
        self.AE_Loss = ContentLoss1(5)
        self.Ad_Loss = torch.nn.BCELoss()

        # 定义优化器
        self.optimizer_AE = optim.Adam(self.AE.parameters(), lr=self.cfg.lr, betas=(self.cfg.b1, self.cfg.b2))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.cfg.lr,
                                      betas=(self.cfg.b1, self.cfg.b2))

    def train_step(self, vi_imgs, epoch, backward=True):
        self.AE.train(backward)
        self.D.train(backward)
        # 生成真假图片的标签
        real_label = (torch.ones(vi_imgs.shape[0], 1) + 0.2*torch.randn(vi_imgs.shape[0], 1)).to(self.device)
        fake_label = (torch.zeros(vi_imgs.shape[0], 1) + 0.3*torch.rand(vi_imgs.shape[0], 1)).to(self.device)

        # 生成图片
        gen_imgs = self.AE(vi_imgs)

        with torch.set_grad_enabled(backward):
            # 定义辨别器损失
            ad_loss = self.Ad_Loss
            real_loss = ad_loss(self.D(vi_imgs), real_label)
            fake_loss = ad_loss(self.D(gen_imgs.detach()), fake_label)
            d_loss = (real_loss + fake_loss) / 2
            # 定义生成器(AE)损失
            pixel_loss = PixelLoss()
            # g_loss = ad_loss(self.D(gen_imgs), real_label) + pixel_loss(gen_imgs, vi_imgs)
            # g_loss = ad_loss(self.D(gen_imgs)
            g_loss = pixel_loss(gen_imgs, vi_imgs)

            # 先训练生成器
            self.optimizer_AE.zero_grad()
            g_loss.backward()
            self.optimizer_AE.step()
            # 再训练辨别器
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            # 保存生成的图片
            if (epoch + 1) % 5 == 0:
                save_image(vi_imgs.data[:25], "../../Data/Vi_imgs/%d.png" % (epoch + 1), nrow=5, normalize=True)
                save_image(gen_imgs.data[:25], "../../Data/Gen_imgs/%d.png" % (epoch + 1), nrow=5, normalize=True)

            return g_loss.item(), d_loss.item()

    @torch.enable_grad()
    def train_over(self, save_dir='../../checkpoint'):
        self.AE.train()
        self.D.train()

        # 准备模型储存路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 准备数据集
        trans = transforms.ToTensor()
        dataset = MyDataset(txt_path=r"../../Data/train.txt", transform=trans)

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # 训练
        for epoch in range(self.cfg.epoch_num):
            for i, (vi_imgs, _) in enumerate(dataloader):
                vi_imgs = vi_imgs.to(self.device)
                (g_loss, d_loss) = self.train_step(vi_imgs, epoch)
                print('Epoch: {} [{}/{}] G_Loss: {:.5f} D_Loss: {:.5f}'.format(
                    epoch + 1, i + 1, len(dataloader), g_loss, d_loss))
                sys.stdout.flush()

            if (epoch + 1) % 5 == 0:
                # 保存模型
                AE_path = os.path.join(save_dir, 'YGC_AE_e%d.pth' % (epoch + 1))
                D_path = os.path.join(save_dir, 'YGC_Dis_e%d.pth' % (epoch + 1))
                torch.save(self.AE.state_dict(), AE_path)
                torch.save(self.D.state_dict(), D_path)

    def parse_args(self, **kwargs):
        # 参数设置
        cfg = {
            'epoch_num': 200,
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
