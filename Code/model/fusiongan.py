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
from Code.model.backbones import Generator_FusionGan, Discriminator_FusionGan
from Code.model.losses import ContentLoss1, PixelLoss
from Code.utils.visual import evaluate_accuracy, Accumulator, Animator, accuracy, evaluate_accuracy1


class FusionGan:
    def __init__(self, G_path=None, D_path=None, **kwargs):
        self.cfg = self.parse_args(**kwargs)

        # GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # 准备模型
        self.G = Generator_FusionGan()
        self.D = Discriminator_FusionGan()
        ops.init_weights(self.G)
        ops.init_weights(self.D)

        # 加载checkpoint
        if G_path is not None:
            self.G.load_state_dict(torch.load(
                G_path, map_location=lambda storage, loc: storage))
        self.G = self.G.to(self.device)

        if D_path is not None:
            self.D.load_state_dict(torch.load(
                D_path, map_location=lambda storage, loc: storage))
        self.D = self.D.to(self.device)

        # 定义损失函数
        self.G_Loss1 = ContentLoss1(5, self.device)
        self.Ad_Loss = torch.nn.BCELoss()

        # 定义优化器
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.cfg.lr, betas=(self.cfg.b1, self.cfg.b2))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.cfg.lr,
                                      betas=(self.cfg.b1, self.cfg.b2))

    def train_step(self, vi_imgs, ir_imgs, epoch, metric, backward=True):
        self.G.train(backward)
        self.D.train(backward)

        # 生成真假图片的标签
        real_label = (torch.ones(vi_imgs.shape[0], 1) + 0.1 * torch.randn(vi_imgs.shape[0], 1)).to(self.device)
        fake_label = (torch.zeros(vi_imgs.shape[0], 1) + 0.3 * torch.rand(vi_imgs.shape[0], 1)).to(self.device)

        # 生成图片
        cat_imgs = torch.cat([vi_imgs, ir_imgs], 1)
        gen_imgs = self.G(cat_imgs)

        with torch.set_grad_enabled(backward):
            # 定义生成器损失
            g_ad_loss = self.Ad_Loss(self.D(gen_imgs), real_label)
            g_ct_loss = self.G_Loss1(vi_imgs, ir_imgs, gen_imgs)
            g_loss = g_ad_loss + g_ct_loss
            # 先训练生成器
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            # 定义辨别器损失
            real_loss = self.Ad_Loss(self.D(vi_imgs), real_label)
            fake_loss = self.Ad_Loss(self.D(gen_imgs.detach()), fake_label)
            d_loss = (real_loss + fake_loss) / 2
            # 再训练辨别器
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            # 计算精度
            Acc = accuracy(self.D(gen_imgs), 0, self.device) + accuracy(self.D(vi_imgs), 1, self.device)
            metric.add(Acc / 2, vi_imgs.shape[0])

            # 保存生成的图片
            if (epoch + 1) % 50 == 0:
                save_image(vi_imgs.data[:25], "../../Data/Vi_imgs/%d.png" % (epoch + 1), nrow=5, normalize=True)
                save_image(ir_imgs.data[:25], "../../Data/Ir_imgs/%d.png" % (epoch + 1), nrow=5, normalize=True)
                save_image(gen_imgs.data[:25], "../../Data/Gen_imgs/%d.png" % (epoch + 1), nrow=5, normalize=True)

            return g_ct_loss.item(), g_ad_loss.item(), d_loss.item()

    @torch.enable_grad()
    def train_over(self, save_dir='../../checkpoint'):
        self.G.train()
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

        # 可视化
        animator = Animator(xlabel='epoch', xlim=[1, self.cfg.epoch_num], ylim=[0.0, 1.0],
                            legend=['train acc', 'test acc'])
        metric = Accumulator(2)

        # 训练
        for epoch in range(self.cfg.epoch_num):
            for i, (vi_imgs, ir_imgs) in enumerate(train_dataloader):
                vi_imgs = vi_imgs.to(self.device)
                ir_imgs = vi_imgs.to(self.device)

                (g_ct_loss, g_ad_loss ,d_loss) = self.train_step(vi_imgs, ir_imgs, epoch, metric)
                print('Epoch: {} [{}/{}] G_CT_Loss: {:.5f} G_AD_Loss: {:.5f} D_Loss: {:.5f}'.format(
                    epoch + 1, i + 1, len(train_dataloader), g_ct_loss ,g_ad_loss , d_loss))
                sys.stdout.flush()


            if (epoch + 1) % 50 == 0:
                # 保存模型
                G_path = os.path.join(save_dir, 'FUSION_GAN_G_e%d.pth' % (epoch + 1))
                D_path = os.path.join(save_dir, 'FUSION_GAN_D_e%d.pth' % (epoch + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)

            if (epoch + 1) % 20 == 0:
                test_acc = evaluate_accuracy1(self.G, self.D, test_dataloader, self.device)
                animator.add(epoch + 1, (metric[0] / metric[1], test_acc))

        # 保存图片
        # plt.show()
        plt.savefig("../../Result/acc")

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
