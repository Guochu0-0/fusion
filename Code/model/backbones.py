from torch import nn


class Generator_FusionGan(nn.Module):
    """
    Fusion-gan中的generator结构，没有分为encoder和decoder
    """

    def __init__(self):
        super(Generator_FusionGan, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x):
        img = self.conv_blocks(x)
        return img


class Discriminator_FusionGan(nn.Module):
    """
    Fusion_gan中的Discriminator结构
    """

    def __init__(self):
        super(Discriminator_FusionGan, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        # 下采样后，特征图的高宽大小，这个地方计算有误，手动修正了
        # ds_size = 120 // 2 ** 4
        ds_size = 8
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class ResidualBlock(nn.Module):
    """
    resnet块，改变激活函数为prelu为了训练gan
    可以选择是否下采样与改变通道数，改变通道数时用1*1卷积改变X的通道数与Z相加
    """

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        if self.conv3:
            x = self.conv3(x)
        return x + residual


class Encoder_YGC(nn.Module):
    """
    随便设计一个网络了，由残差块构成
    """

    def __init__(self):
        super(Encoder_YGC, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64, 64)
        self.block3 = ResidualBlock(64, 64)
        self.block4 = ResidualBlock(64, 64)
        self.block5 = ResidualBlock(64, 64)
        self.block6 = ResidualBlock(64, 64)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)

        return block6

class Decoder_YGC(nn.Module):
    """
    随便设计一个网络了，由残差块构成
    """

    def __init__(self):
        super(Decoder_YGC, self).__init__()

        self.block1 = ResidualBlock(64, 64)
        self.block2 = ResidualBlock(64, 64)
        self.block3 = ResidualBlock(64, 64)
        self.block4 = ResidualBlock(64, 64)
        self.block5 = ResidualBlock(64, 64)
        self.block6 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)

        return block6

class Decoder_YGC_V2(nn.Module):
    """
    解码器，容量设小一点
    """

    def __init__(self):
        super(Decoder_YGC, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)

        return block6

class AE_YGC(nn.Module):
    def __init__(self):
        super(AE_YGC, self).__init__()
        self.E = Encoder_YGC()
        self.D = Decoder_YGC()

    def forward(self, x):
        z = self.E(x)
        out = self.D(z)

        return out