import tensorflow as tf

__all__ = ["DABNet"]

def Conv(input, nOut_channels, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
    output = tf.layers.conv2d(input,
                                 filters=nOut_channels,
                                 kernel_size=kSize,
                                 strides=stride,
                                 dilation_rate=dilation,
                                 padding=padding,
                                 use_bias=bias)
    if bn_acti:
        output = BNPReLU(output)
    return output


def BNPReLU(input):

    output = tf.layers.BatchNormalization(input, epsilon=1e-3)
    output = tf.nn.relu(output)
    return output

def DABModule(input, , out_channels, d=1, kSize=3, dkSize=3):

    bn_relu_1 = BNPReLU(input)
    conv3x3 = Conv(bn_relu_1, out_channels, kSize, 1, padding=1,
                   bn_acti=True)
    dconv3x1 = Conv(conv3x3, conv3x3.shape[-1] // 2, (dkSize, 1), 1,
                    padding=(1, 0), groups=conv3x3.shape[-1] // 2,
                    bn_acti=True)
    dconv1x3 = Conv(dconv3x1, dconv3x1.shape[-1] // 2, (1, dkSize), 1,
                    padding=(0, 1), groups=dconv3x1.shape[-1] // 2, bn_acti=True)
    ddconv3x1 = Conv(conv3x3, conv3x3.shape[-1] // 2, (dkSize, 1), 1,
                     padding=(1 * d, 0), dilation=(d, 1),
                     groups=conv3x3.shape[-1] // 2, bn_acti=True)
    ddconv1x3 = Conv(ddconv3x1, ddconv3x1.shape[-1] // 2, (1,dkSize), 1,
                     padding=(0, 1 * d), dilation=(1, d),
                     groups=ddconv3x1.shape[-1] // 2, bn_acti=True)
    output = tf.concat(dconv1x3, ddconv1x3)
    bn_relu_2 = BNPReLU(output)
    conv1x1 = Conv(bn_relu_2, bn_relu_2.shape[-1], 1, 1, padding=0,
                   bn_acti=False)
    return tf.concat(conv1x1 , input)


def DownSamplingBlock(input, nOut_channel):
    if input.shape[-1] < nOut_channel:
        nConv = nOut_channel - input.shape[-1]
    else:
        nConv = nOut_channel
    output = Conv(input, nConv, kSize=3, stride=2, padding="same")
    if input.shape[-1] < nOut_channel:
        max_pool = tf.layers.max_pooling2d(input)
        output = tf.concat(output,max_pool)
    output = BNPReLU(output)
    return output


def InputInjection(input, ratio):
    for i in range(0, ratio):
        input = tf.layers.average_pooling2d(input, pool_size=(3,3), strides=2, padding="same")
    return input


class DABNet():
    def __init__(self, classes=2, block_1=3, block_2=6):
        super().__init__()
        self.classes = classes
        self.block_1 = block_1
        self.block_2 = block_2
        # self.init_conv = nn.Sequential(
        #     Conv(3, 32, 3, 2, padding=1, bn_acti=True),
        #     Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        #     Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        # )
        #
        # self.down_1 = InputInjection(1)  # down-sample the image 1 times
        # self.down_2 = InputInjection(2)  # down-sample the image 2 times
        # self.down_3 = InputInjection(3)  # down-sample the image 3 times
        #
        # self.bn_prelu_1 = BNPReLU(32 + 3)
        #
        # # DAB Block 1
        #
        # self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        # self.DAB_Block_1 = nn.Sequential()
        # for i in range(0, block_1):
        #     self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        # self.bn_prelu_2 = BNPReLU(128 + 3)
        #
        # # DAB Block 2
        # dilation_block_2 = [4, 4, 8, 8, 16, 16]
        # self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        # self.DAB_Block_2 = nn.Sequential()
        # for i in range(0, block_2):
        #     self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
        #                                 DABModule(128, d=dilation_block_2[i]))
        # self.bn_prelu_3 = BNPReLU(256 + 3)
        #
        # self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):
        output0 =  Conv(input, 32, 3, 2, padding=1, bn_acti=True),
        output0 = Conv(output0, 32, 3, 1, padding=1, bn_acti=True),
        output0 = Conv(output0, 32, 3, 1, padding=1, bn_acti=True),

        down_1 =  InputInjection(output0, 1)  # down-sample the image 1 times
        down_2 = InputInjection(down_1, 2)
        down_3 = InputInjection(down_2, 3)

        output0_cat = BNPReLU(tf.concat(output0, down_1))
        # DAB Block 1
        output1_0 = DownSamplingBlock(output0_cat, 64)
        output1 = output1_0
        for i in range(0, self.block_1):
         # self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
            output1 = DABModule(output1, 32, d=2)

        output1_cat = BNPReLU(tf.concat(output1, output1_0, down_2))
        # DAB Block 2
        output2_0 = DownSamplingBlock(output1_cat, 128)
        output2 = output2_0
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        for i in range(0, self.block_2):
            DABModule(output2, d=dilation_block_2[i])
        output2_cat = BNPReLU(tf.concat(output2, output2_0, down_3))
        out = Conv(output2_cat, self.classes, 1, 1, padding=0)
        out = tf.image.resize_nearest_neighbor(out, input.size()[2:])
        # output0 = self.init_conv(input)
        #
        # down_1 = self.down_1(input)
        # down_2 = self.down_2(input)
        # down_3 = self.down_3(input)
        #
        # output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        # # DAB Block 1
        # output1_0 = self.downsample_1(output0_cat)
        # output1 = self.DAB_Block_1(output1_0)
        # output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))
        #
        # # DAB Block 2
        # output2_0 = self.downsample_2(output1_cat)
        # output2 = self.DAB_Block_2(output2_0)
        # output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))
        #
        # out = self.classifier(output2_cat)
        # out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)
        return out

"""print layers and params of network"""
if __name__ == '__main__':

    model = DABNet(classes=2)

