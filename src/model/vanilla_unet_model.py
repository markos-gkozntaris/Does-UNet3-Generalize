from .vanilla_unet_submodules import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, pad):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pad = pad

        self.dconv = DoubleConv(n_channels, 64) #n_chanels is the number of input channels, while the number of output channels here is 64
        self.down1 = Down(64, 128)              # input and output channels can be cheked at Fig. 1 of the original paper
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        #
        self.up1 = Up(1024, 512,pad)
        self.up2 = Up(512, 256,pad)
        self.up3 = Up(256, 128,pad)
        self.up4 = Up(128, 64,pad)
        self.outc = OutConv(64, n_classes)

    def forward(self, image):
        # expected input image (batch size, channels, height, width)
        # encoder
        x1 = self.dconv(image)
        print("x1 = ", x1.size())
        x2 = self.down1(x1)
        print("x2 = ", x2.size())
        x3 = self.down2(x2)
        print("x3 = ", x3.size())
        x4 = self.down3(x3)
        print("x4 = ", x4.size())
        x5 = self.down4(x4)
        print("x5 = ", x5.size())        

        # decoder
        image = self.up1(x5, x4)
        print("after up 1 = ", image.size())
        image = self.up2(image, x3)
        print("after up 2 = ", image.size())
        image = self.up3(image, x2)
        print("after up 3 = ", image.size())
        image = self.up4(image, x1)
        print("after up 4 = ", image.size())
        # final pass with kernel 1x1 
        semantic_segment = self.outc(image)
        print("output image size = ",semantic_segment.size())
        return semantic_segment


