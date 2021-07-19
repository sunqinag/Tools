import torch.nn as nn
import torch
import torch.nn.functional as F

#bz unet
class unet_conv_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding_pool=0,
                 padding_conv=1,
                 kernel_size=3,
                 stride=1):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding_conv),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding_conv),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2,padding=padding_pool)

    def forward(self, x):
        conv_a = self.conv_a(x)
        conv_b = self.conv_b(conv_a)
        maxpool = self.maxpool(conv_b)
        return conv_a,conv_b,maxpool

class unet_deconv_block(nn.Module):
    def __init__(self,in_channels, out_channels, padding=1, kernel_size=3,
                 stride=2, output_padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                               padding, output_padding),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.deconv_block(x)

class Concat_fuse_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1):
        super().__init__()
        self.concat_fuse=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.concat_fuse(x)

class DOWN(nn.Module):
    def __init__(self,conv_filter_nums):
        super().__init__()
        self.down1 = unet_conv_block(conv_filter_nums[0], conv_filter_nums[1])
        self.down2 = unet_conv_block(conv_filter_nums[1], conv_filter_nums[2])
        self.down3 = unet_conv_block(conv_filter_nums[2], conv_filter_nums[3], 1)
        self.down4 = unet_conv_block(conv_filter_nums[3], conv_filter_nums[4], 1)
        self.down5 = unet_conv_block(conv_filter_nums[4], conv_filter_nums[5])

    def forward(self, x):
        conv1_1, conv1_2, maxpool1 = self.down1(x)
        conv2_1, conv2_2, maxpool2 = self.down2(maxpool1)
        conv3_1, conv3_2, maxpool3 = self.down3(maxpool2)
        conv4_1, conv4_2, maxpool4 = self.down4(maxpool3)
        conv5_1, conv5_2, maxpool5 = self.down5(maxpool4)
        return (conv1_1, conv1_2,conv2_1, conv2_2,conv3_1, conv3_2,conv4_1,
                conv4_2, conv5_1, conv5_2),\
               (maxpool1,maxpool2,maxpool3,maxpool4,maxpool5)

class UP(nn.Module):
    def __init__(self,deconv_filter_nums):
        super().__init__()
        self.deconv_filter_nums=deconv_filter_nums #[256,128,64,32,16,8]
        self.unet_deconv_block_0 = unet_deconv_block(self.deconv_filter_nums[0],
                                                     self.deconv_filter_nums[1])
        self.conv_concat_0 = Concat_fuse_block(768, self.deconv_filter_nums[1])
        self.unet_deconv_block_1 = unet_deconv_block(self.deconv_filter_nums[1],
                                                     self.deconv_filter_nums[2])
        self.conv_concat_1 = Concat_fuse_block(384, self.deconv_filter_nums[2],
                                                     1, 1, )
        self.unet_deconv_block_2 = unet_deconv_block(self.deconv_filter_nums[2],
                                                     self.deconv_filter_nums[3])
        self.conv_concat_2 = Concat_fuse_block(192, self.deconv_filter_nums[3],
                                                     1, 1, )
        self.unet_deconv_block_3 = unet_deconv_block(self.deconv_filter_nums[3],
                                                     self.deconv_filter_nums[4])
        self.conv_concat_3 = Concat_fuse_block(96, self.deconv_filter_nums[4],
                                                     1, 1, )
        self.unet_deconv_block_4 = unet_deconv_block(self.deconv_filter_nums[4],
                                                     self.deconv_filter_nums[5])

    def forward(self, x,conv_list,pool_list):
        deconv_0 = self.unet_deconv_block_0(pool_list[4])
        deconv_interpolate_0 = F.interpolate(deconv_0, size=(pool_list[3].size()[2],
                   pool_list[3].size()[3]), mode='bilinear',align_corners=False)
        concat_0 = torch.cat([conv_list[8], conv_list[9], pool_list[3],
                              deconv_interpolate_0], 1)
        conv_concat_0 = self.conv_concat_0(concat_0)

        deconv_1 = self.unet_deconv_block_1(conv_concat_0)
        deconv_interpolate_1 = F.interpolate(deconv_1, size=(pool_list[2].size()[2],
                pool_list[2].size()[3]), mode='bilinear',align_corners=False)
        concat_1 = torch.cat([conv_list[6], conv_list[7],
                                    pool_list[2], deconv_interpolate_1], 1)
        conv_concat_1 = self.conv_concat_1(concat_1)

        deconv_2 = self.unet_deconv_block_2(conv_concat_1)
        deconv_interpolate_2 = F.interpolate(deconv_2, size=(pool_list[1].size()[2],
                    pool_list[1].size()[3]), mode='bilinear',align_corners=False)
        concat_2 = torch.cat([conv_list[4], conv_list[5], pool_list[1],
                                                    deconv_interpolate_2], 1)
        conv_concat_2 = self.conv_concat_2(concat_2)

        deconv_3 = self.unet_deconv_block_3(conv_concat_2)
        deconv_interpolate_3 = F.interpolate(deconv_3, size=(pool_list[0].size()[2],
                    pool_list[0].size()[3]), mode='bilinear',align_corners=False)
        concat_3 = torch.cat([conv_list[2], conv_list[3], pool_list[0],
                                                    deconv_interpolate_3], 1)
        conv_concat_3 = self.conv_concat_3(concat_3)

        deconv_4 = self.unet_deconv_block_4(conv_concat_3)
        deconv_interpolate_4 = F.interpolate(deconv_4, size=(x.size()[2],
                            x.size()[3]), mode='bilinear',align_corners=False)
        return deconv_interpolate_4

class bz_Unet(nn.Module):
    def __init__(self,channels,num_classes):
        super().__init__()
        self.num_classes=num_classes
        self.conv_filter_nums=[channels,16,32,64,128,256]
        self.deconv_filter_nums=[256,128,64,32,16,8]
        self.down=DOWN(conv_filter_nums=self.conv_filter_nums)#卷积
        self.up=UP(deconv_filter_nums=self.deconv_filter_nums)#反卷积
        self.outconv=nn.Conv2d(self.deconv_filter_nums[-1], self.num_classes, 1, 1)

    def forward(self,x):
        conv_list,pool_list=self.down(x)
        deconv_interpolate=self.up(x,conv_list,pool_list)
        out_conv=self.outconv(deconv_interpolate)
        return out_conv

def bz_unet(channels,num_classes):
    return bz_Unet(channels,num_classes)

#git unet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2,
                                                        kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/\
        # commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/\
        # 8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def unet(channels,num_classes):
    return UNet(channels,num_classes)