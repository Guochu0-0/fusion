import os
import sys

import torch
from h5py.h5d import namedtuple
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.utils import save_image

from Code.data.mydataset import MyDataset
from Code.model import ops
from Code.model.backbones import Discriminator_FusionGan, AE_YGC
from Code.model.losses import ContentLoss1, PixelLoss
from Code.utils.visual import accuracy, Accumulator, Animator, evaluate_accuracy


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

        # 可视化
        self.writer = SummaryWriter(os.path.join("..", "..", "Results", "rebuild"))

    def train_step(self, vi_imgs, epoch, metric, backward=True):
        self.AE.train(backward)
        self.D.train(backward)
        # 生成真假图片的标签
        real_label = (torch.ones(vi_imgs.shape[0], 1) + 0.1 * torch.randn(vi_imgs.shape[0], 1)).to(self.device)
        fake_label = (torch.zeros(vi_imgs.shape[0], 1) + 0.3 * torch.rand(vi_imgs.shape[0], 1)).to(self.device)

        # 生成图片
        gen_imgs = self.AE(vi_imgs)

        with torch.set_grad_enabled(backward):
            # 定义生成器(AE)损失
            ad_loss = self.Ad_Loss
            pixel_loss = PixelLoss()
            # g_loss = ad_loss(self.D(gen_imgs), real_label) + pixel_loss(gen_imgs, vi_imgs)
            g_loss = ad_loss(self.D(gen_imgs), real_label)
            # g_loss = pixel_loss(gen_imgs, vi_imgs)
            # 先训练生成器
            self.optimizer_AE.zero_grad()
            g_loss.backward()
            self.optimizer_AE.step()

            # 定义辨别器损失
            real_loss = ad_loss(self.D(vi_imgs), real_label)
            fake_loss = ad_loss(self.D(gen_imgs.detach()), fake_label)
            d_loss = (real_loss + fake_loss) / 2
            # 再训练辨别器
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            # 可视化loss
            self.writer.add_scalars("loss", {"g_loss": g_loss,
                                             "d_loss": d_loss}, (epoch + 1))

            # 计算精度
            Acc = accuracy(self.D(gen_imgs), 0, self.device) + accuracy(self.D(vi_imgs), 1, self.device)
            metric.add(Acc / 2, vi_imgs.shape[0])

            # 可视化生成的图片
            if (epoch + 1) % 50 == 0:
                vi_imgs_show = vutils.make_grid(vi_imgs)
                gen_imgs_show = vutils.make_grid(gen_imgs)
                self.writer.add_image("vi_imgs", vi_imgs_show, (epoch + 1))
                self.writer.add_image("gen_imgs", gen_imgs_show, (epoch + 1))

            return g_loss.item(), d_loss.item()

    @torch.enable_grad()
    def train_over(self, save_dir='../../checkpoint'):
        self.AE.train()
        self.D.train()

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

        test_trans = transforms.ToTensor()
        test_dataset = MyDataset(txt_path=r"../../Data/test.txt", transform=test_trans)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # 训练
        for epoch in range(self.cfg.epoch_num):
            # 可视化
            metric = Accumulator(2)

            for i, (vi_imgs, _) in enumerate(train_dataloader):
                vi_imgs = vi_imgs.to(self.device)
                (g_loss, d_loss) = self.train_step(vi_imgs, epoch, metric)
                print('Epoch: {} [{}/{}] G_Loss: {:.5f} D_Loss: {:.5f}'.format(
                    epoch + 1, i + 1, len(train_dataloader), g_loss, d_loss))
                sys.stdout.flush()

            if (epoch + 1) % 50 == 0:
                # 保存模型
                AE_path = os.path.join(save_dir, 'YGC_AE_e%d.pth' % (epoch + 1))
                D_path = os.path.join(save_dir, 'YGC_Dis_e%d.pth' % (epoch + 1))
                torch.save(self.AE.state_dict(), AE_path)
                torch.save(self.D.state_dict(), D_path)

            if (epoch + 1) % 20 == 0:
                test_acc = evaluate_accuracy(self.AE, self.D, test_dataloader, self.device)
                self.writer.add_scalars("acc", {"test_acc": test_acc,
                                                "train_acc": metric[0] / metric[1]}, (epoch+1))

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
