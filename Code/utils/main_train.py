import argparse

from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch

from Code.data.mydataset import MyDataset
from Code.model.fusiongan import Generator, Discriminator, weights_init_normal
from Code.model.loss import ContentLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    # 准备模型
    generator = Generator()
    discriminator = Discriminator()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    adversarial_loss = torch.nn.BCELoss()
    content_loss = ContentLoss(5)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 数据
    train_data = MyDataset(txt_path=r"../../Data/train.txt")
    dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

    # GPU
    cuda = True if torch.cuda.is_available() else False

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        content_loss.cuda()
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    # 训练
    for epoch in range(opt.n_epochs):
        for i, (vi_imgs, ir_imgs) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(vi_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(vi_imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            # 将可见光图片和红外图片按通道连接
            cat_imgs = torch.cat([vi_imgs, ir_imgs], 1)
            gen_imgs = generator(cat_imgs)

            ad_g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            ct_g_loss = content_loss(vi_imgs, ir_imgs, gen_imgs)
            g_loss = ad_g_loss + ct_g_loss

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(vi_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f][ad G loss: %f][ct G loss: %f]"
                % (
                    epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), ad_g_loss.item(),
                    ct_g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(vi_imgs.data[:25], "../../Data/Vi_imgs/%d.png" % batches_done, nrow=5, normalize=True)
                save_image(gen_imgs.data[:25], "../../Data/Gen_imgs/%d.png" % batches_done, nrow=5, normalize=True)
