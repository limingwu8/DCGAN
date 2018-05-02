import torch.nn as nn
from utils import Opt


def up_block(in_channels, out_channels, kernel_size=4, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def up_output(in_channels, out_channels, kernel_size=4, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.Tanh()
    )

def down_block(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def down_output(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.Sigmoid()
    )

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)

### for 32x32, works
# class netG(nn.Module):
#     def __init__(self):
#         super(netG, self).__init__()
#         self.up_block1 = up_block(Opt.nz, Opt.ngf * 8, 4, 1, 0)
#         self.up_block2 = up_block(Opt.ngf * 8, Opt.ngf * 4, 4, 2, 1)
#         self.up_block3 = up_block(Opt.ngf * 4, Opt.ngf * 2, 4, 2, 1)
#         self.up_block4 = up_block(Opt.ngf * 2, Opt.nc, 4, 2, 1)
#         # self.up_block5 = up_output(Opt.ngf, Opt.nc, 2, 2, 0)
#
#     def forward(self, input):
#         x1 = self.up_block1(input)
#         x2 = self.up_block2(x1)
#         x3 = self.up_block3(x2)
#         x4 = self.up_block4(x3)
#         # x5 = self.up_block5(x4)
#
#         return x4
#
# class netD(nn.Module):
#     def __init__(self):
#         super(netD, self).__init__()
#         self.down_block1 = down_block(Opt.nc, Opt.ndf, 2, 2, 0)
#         self.down_block2 = down_block(Opt.ndf, Opt.ndf * 2, 2, 2, 0)
#         self.down_block3 = down_block(Opt.ndf * 2, Opt.ndf * 4, 2, 2, 0)
#         self.down_block4 = down_block(Opt.ndf * 4, Opt.ndf * 8, 2, 2, 0)
#         self.down_block5 = down_output(Opt.ndf * 8, 1, 2, 2, 0)
#
#     def forward(self, input):
#         x1 = self.down_block1(input)
#         x2 = self.down_block2(x1)
#         x3 = self.down_block3(x2)
#         x4 = self.down_block4(x3)
#         x5 = self.down_block5(x4)
#         output = x5.view(-1, 1).squeeze(-1)
#         return output

## for 64x64, works
class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.up_block1 = up_block(Opt.nz, 1024, 4, 1, 0)
        self.up_block2 = up_block(1024, 512, 4, 2, 1)
        self.up_block3 = up_block(512, 256, 4, 2, 1)
        self.up_block4 = up_block(256, 128, 4, 2, 1)
        self.up_block5 = up_output(128, Opt.nc, 4, 2, 1)

    def forward(self, input):
        x1 = self.up_block1(input)
        x2 = self.up_block2(x1)
        x3 = self.up_block3(x2)
        x4 = self.up_block4(x3)
        x5 = self.up_block5(x4)

        return x5

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.down_block1 = down_block(Opt.nc, 128, 4, 2, 1)
        self.down_block2 = down_block(128, 256, 4, 2, 1)
        self.down_block3 = down_block(256, 512, 4, 2, 1)
        self.down_block4 = down_block(512, 1024, 4, 2, 1)
        self.down_block5 = down_output(1024, 1, 4, 1, 0)

    def forward(self, input):
        x1 = self.down_block1(input)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        output = x5.view(-1, 1).squeeze(-1)
        return output


### for 64x64, does not work well
# class netG(nn.Module):
#     def __init__(self):
#         super(netG, self).__init__()
#         self.up_block1 = up_block(Opt.nz, 1024, 4, 4, 0)
#         self.up_block2 = up_block(1024, 512, 4, 4, 4)
#         self.up_block3 = up_block(512, 256, 4, 4, 8)
#         self.up_block4 = up_block(256, 128, 4, 4, 16)
#         self.up_block5 = up_output(128, 1, 4, 4, 32)
#
#     def forward(self, input):
#         x1 = self.up_block1(input)
#         x2 = self.up_block2(x1)
#         x3 = self.up_block3(x2)
#         x4 = self.up_block4(x3)
#         x5 = self.up_block5(x4)
#
#         return x5
#
# class netD(nn.Module):
#     def __init__(self):
#         super(netD, self).__init__()
#         self.down_block1 = down_block(Opt.nc, 128, 4, 2, 1)
#         self.down_block2 = down_block(128, 256, 4, 2, 1)
#         self.down_block3 = down_block(256, 512, 4, 2, 1)
#         self.down_block4 = down_block(512, 1024, 4, 2, 1)
#         self.down_block5 = down_output(1024, 1, 4, 1, 0)
#
#     def forward(self, input):
#         x1 = self.down_block1(input)
#         x2 = self.down_block2(x1)
#         x3 = self.down_block3(x2)
#         x4 = self.down_block4(x3)
#         x5 = self.down_block5(x4)
#         output = x5.view(-1, 1).squeeze(-1)
#         return output