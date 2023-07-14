import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from antialiased_cnns.blurpool import BlurPool

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=32, normal=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,      下采样的次数（每次对半）
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(UnetGenerator, self).__init__()
        self.normal = normal
        self.e1 = nn.Sequential(*[
            nn.Conv2d(input_nc, 32, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
        ])
        self.e2 = nn.Sequential(*[
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(64, stride=2)
        ])
        self.e3 = nn.Sequential(*[
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(128, stride=2)
        ])
        self.e4 = nn.Sequential(*[
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(256, stride=2)
        ])
        self.e5 = nn.Sequential(*[
            nn.Conv2d(256, 512, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(512, stride=2)
        ])
        self.e6 = nn.Sequential(*[
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(512, stride=2)
        ])
        self.bottle = nn.Sequential(*[
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True),
            BlurPool(512, stride=2)
        ])
        self.d1 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d2 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d3 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d4 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d5 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d6 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])


        if normal:
            self.d5_l = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(256, 64, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True)
            ])
            self.d6_l = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(128, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True)
            ])

            self.out = nn.Sequential(*[
                nn.Conv2d(64, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, output_nc // 2, 1, 1)
            ])
            self.out_l = nn.Sequential(*[
                nn.Conv2d(64, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, output_nc // 2, 1, 1)
            ])
        else:
            self.out = nn.Sequential(*[
                nn.Conv2d(32, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, output_nc, 1, 1),
                # nn.ReLU()
            ])


    def forward(self, x):
        o1 = self.e1(x)
        o2 = self.e2(o1)
        o3 = self.e3(o2)
        o4 = self.e4(o3)
        o5 = self.e5(o4)
        o6 = self.e6(o5)
        bottle = self.bottle(o6)
        d1 = self.d1(bottle)
        d2 = self.d2(torch.cat([o6, d1], 1))
        d3 = self.d3(torch.cat([o5, d2], 1))
        d4 = self.d4(torch.cat([o4, d3], 1))
        d5 = self.d5(torch.cat([o3, d4], 1))
        d6 = self.d6(torch.cat([o2, d5], 1))

        if self.normal:
            d5_l = self.d5_l(torch.cat([o3, d4], 1))
            d6_l = self.d6_l(torch.cat([o2, d5_l], 1))

            out = self.out(torch.cat([o1, d6], 1))
            out_l = self.out_l(torch.cat([o1, d6_l], 1))

            return torch.cat([out, out_l], 1)

        out = self.out(d6)

        return out


class specular(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=8):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,      下采样的次数（每次对半）
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(specular, self).__init__()

        self.e1 = nn.Sequential(*[
            nn.Conv2d(input_nc, 8, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
        ])
        self.e2 = nn.Sequential(*[
            nn.Conv2d(8, 16, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(16, stride=2)
        ])
        self.e3 = nn.Sequential(*[
            nn.Conv2d(16, 32, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(32, stride=2)
        ])
        self.e4 = nn.Sequential(*[
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(64, stride=2)
        ])
        self.e5 = nn.Sequential(*[
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(128, stride=2)
        ])
        self.e6 = nn.Sequential(*[
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(256, stride=2)
        ])
        self.bottle = nn.Sequential(*[
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True),
            BlurPool(256, stride=2)
        ])
        self.d1 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d2 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d3 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d4 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d5 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d6 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.out = nn.Sequential(*[
            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        

        self.out1 = nn.Sequential(*[
            nn.Conv2d(16, output_nc, 1, 1),
            # nn.Softmax()
            # nn.ReLU(True)
            # nn.Sigmoid()
        ])



    def forward(self, x):
        o1 = self.e1(x)
        o2 = self.e2(o1)
        o3 = self.e3(o2)
        o4 = self.e4(o3)
        o5 = self.e5(o4)
        o6 = self.e6(o5)
        bottle = self.bottle(o6)
        d1 = self.d1(bottle)
        d2 = self.d2(torch.cat([o6, d1], 1))
        d3 = self.d3(torch.cat([o5, d2], 1))
        d4 = self.d4(torch.cat([o4, d3], 1))
        d5 = self.d5(torch.cat([o3, d4], 1))
        d6 = self.d6(torch.cat([o2, d5], 1))
        out = self.out(torch.cat([o1, d6], 1))

        out = self.out1(out)

        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=32, norm='Batch', use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0) # 若条件满足则继续往下运行，否则显示AssertError，停止往下运行。
        super(ResnetGenerator, self).__init__()
        use_bias = True
        norm_layer = nn.BatchNorm2d
        model = [nn.ReflectionPad2d(3),   # 上下左右均填充三行像素，最终就是行填充了6行，列填充了6列
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        # model += [nn.ReLU(True)]
        model += [nn.LeakyReLU(0.7)]

        self.model = nn.Sequential(*model) # 将字典类的model转换为真正的可以运行的model

    def forward(self, x): 
        return self.model(x)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a VGG discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        use_bias = True

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.linear = nn.Linear(ndf * nf_mult * 32 * 32, 1, bias=True)  # output 1 channel prediction map
        self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        out = self.model(input).reshape(input.shape[0], -1)
        logits = self.linear(out)
        sigmoid = self.sigmoid(logits)
        return logits, sigmoid


class GANLoss(nn.Module):  # 继承nn.Module时，需要用到super函数；数据集继承ABC时不用。
    def __init__(self, mask, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.mask = mask
        self.gan_type = gan_type.lower() # 全部转为小写
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val) # 返回与输入同样大小的tensor，其中所有值都置为self.real_label_val
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss



def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, local_rank, init_type='normal', init_gain=0.02):
    device = torch.device('cuda', local_rank)
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, local_rank, normal=False, init_type='normal', init_gain=0.02, Net="Unet"):
    if Net == 'Unet':
        net = UnetGenerator(input_nc, output_nc, ngf, normal=normal)
    elif Net == "Res":
        net = ResnetGenerator(input_nc, output_nc, ngf, use_dropout=False)
    else:
        net = specular(input_nc, output_nc, ngf)
    return init_net(net, local_rank, init_type, init_gain)


def define_D(input_nc, ngf, local_rank, init_type='normal', init_gain=0.02):
    net = None
    net = NLayerDiscriminator(input_nc, ngf)
    return init_net(net, local_rank, init_type, init_gain)
