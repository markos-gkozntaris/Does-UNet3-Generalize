import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.unet3_submodules import *
from .init_weights import init_weights


class UNet3Plus(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, feature_scale=4,
                 is_deconv=True, is_batchnorm=True, print = True):
        super(UNet3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.print = print
        filters = [64, 128, 256, 512, 1024]

        ### -------------Encoder-------------- ###
        
        """Encoder side is the same as that of simple unet"""
        self.dconv = DoubleConv(self.n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1]) 
        self.down2 = Down(filters[1], filters[2]) 
        self.down3 = Down(filters[2], filters[3])
        self.down4 = Down(filters[3], filters[4])   

        ### -------------Decoder-------------- ###
        
        self.CatChannels = filters[0]       # My comment: output channels, number of channels required for Concatination
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks     # each output of the decoder block will have a size of filter[0]*5, because it has 5 inputs of the same dimesnions to be created

        '''create X_DE^4'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_de4_Conv_BN_ReLU = Conv_BN_ReLU(filters[0],self.CatChannels)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_de4_Conv_BN_ReLU = Conv_BN_ReLU(filters[1],self.CatChannels)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_de4_Conv_BN_ReLU = Conv_BN_ReLU(filters[2],self.CatChannels)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_PT_de4_Conv_BN_ReLU = Conv_BN_ReLU(filters[3],self.CatChannels)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.h5_PT_de4_Conv_BN_ReLU = Conv_BN_ReLU(filters[4],self.CatChannels)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.de4_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.UpChannels)

        '''create X_DE^3'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_de3_Conv_BN_ReLU = Conv_BN_ReLU(filters[0],self.CatChannels)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_de3_Conv_BN_ReLU = Conv_BN_ReLU(filters[1],self.CatChannels)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_PT_de3_Conv_BN_ReLU = Conv_BN_ReLU(filters[2],self.CatChannels)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.h4_PT_de3_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.CatChannels)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.h5_PT_de3_Conv_BN_ReLU = Conv_BN_ReLU(filters[4],self.CatChannels)     # decoder number  comes directly from the two convolutions without upsampling so the number of its channels will come from filter array

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.de3_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.UpChannels)

        '''create X_DE^2'''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_de2_Conv_BN_ReLU = Conv_BN_ReLU(filters[0],self.CatChannels)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_PT_de2_Conv_BN_ReLU = Conv_BN_ReLU(filters[1],self.CatChannels)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_PT_de2_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.CatChannels)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_PT_de2_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels, self.CatChannels)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_PT_de2_Conv_BN_ReLU = Conv_BN_ReLU(filters[4], self.CatChannels)


        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.de2_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.UpChannels)

        '''create X_DE^1'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_PT_de1_Conv_BN_ReLU = Conv_BN_ReLU(filters[0],self.CatChannels)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_PT_de1_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.CatChannels)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_PT_de1_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.CatChannels)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_PT_de1_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.CatChannels)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_PT_de1_Conv_BN_ReLU = Conv_BN_ReLU(filters[4],self.CatChannels)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.de1_Conv_BN_ReLU = Conv_BN_ReLU(self.UpChannels,self.UpChannels)


        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        ### -------------Encoder------------- ###
        # same as Unet vanilla
        x1 = self.dconv(inputs)     # h1->320*320*64
        x2 = self.down1(x1)         # h2->160*160*128
        x3 = self.down2(x2)         # h3->80*80*256
        x4 = self.down3(x3)         # h4->40*40*512
        hde5 = self.down4(x4)        # h5->20*20*1024

        if self.print:
            print("x1 = ", x1.size())
            print("x2 = ", x2.size())
            print("x3 = ", x3.size())
            print("x4 = ", x4.size())
            print("x5 = ", hde5.size())

        ### -------------Decoder------------- ###

        '''create X_DE^4'''
        # temp = self.h1_PT_hd4(x1)
        h1_PT_hd4 = self.h1_PT_de4_Conv_BN_ReLU(self.h1_PT_hd4(x1))
        h2_PT_hd4 = self.h2_PT_de4_Conv_BN_ReLU(self.h2_PT_hd4(x2))
        h3_PT_hd4 = self.h3_PT_de4_Conv_BN_ReLU(self.h3_PT_hd4(x3))
        h4_Cat_hd4 = self.h4_PT_de4_Conv_BN_ReLU(x4)
        p2d = (0, 1, 0, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        h4_Cat_hd4 = F.pad(h4_Cat_hd4, p2d, "constant", 0)
        hd5_UT_hd4 = self.h5_PT_de4_Conv_BN_ReLU(self.hd5_UT_hd4(hde5))
        p3d = (1, 1, 1, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        hd5_UT_hd4 = F.pad(hd5_UT_hd4, p3d, "constant", 0)
        if self.print:
            # print("temp = ", temp.size())
            print("h1_PT_hd4 = ", h1_PT_hd4.size())
            print("h2_PT_hd4 = ", h2_PT_hd4.size())
            print("h3_PT_hd4 = ", h3_PT_hd4.size())
            print("h4_Cat_hd4 = ", h4_Cat_hd4.size())
            print("hd5_UT_hd4 = ", hd5_UT_hd4.size())

        hde4 = self.de4_Conv_BN_ReLU(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)) # hd4->40*40*UpChannels
        print("hde4 = ", hde4.size())
        
        

        '''create X_DE^3'''
        h1_PT_hd3 = self.h1_PT_de3_Conv_BN_ReLU(self.h1_PT_hd3(x1))
        # TODO:
        p2d = (0, 1, 0, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        h1_PT_hd3 = F.pad(h1_PT_hd3, p2d, "constant", 0)
        print("h1_PT_hd3 = ", h1_PT_hd3.size())

        h2_PT_hd3 = self.h2_PT_de3_Conv_BN_ReLU(self.h2_PT_hd3(x2))
        p2d = (0, 1, 0, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        h2_PT_hd3 = F.pad(h2_PT_hd3, p2d, "constant", 0)
        print("h2_PT_hd3 = ", h2_PT_hd3.size())

        h3_Cat_hd3 = self.h3_PT_de3_Conv_BN_ReLU(x3)
        p2d = (0, 1, 0, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        h3_Cat_hd3 = F.pad(h3_Cat_hd3, p2d, "constant", 0)
        print("h3_Cat_hd3 = ", h3_Cat_hd3.size())

        hd4_UT_hd3 = self.h4_PT_de3_Conv_BN_ReLU(self.hd4_UT_hd3(hde4))     # the one that come from decoder of stage 4  
        print("hd4_UT_hd3 = ", hd4_UT_hd3.size())

        hd5_UT_hd3 = self.h5_PT_de3_Conv_BN_ReLU(self.hd5_UT_hd3(hde5)) 
        p2d = (2, 2, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        hd5_UT_hd3 = F.pad(hd5_UT_hd3, p2d, "constant", 0)    # the one that come from decoder of stage 5
        print("hd5_UT_hd3 = ", hd5_UT_hd3.size())
        
        hde3 = self.de3_Conv_BN_ReLU(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)) # hd3->80*80*UpChannels

        if self.print:
            # print("temp = ", temp.size())
            print("h1_PT_hd4 = ", h1_PT_hd3.size())
            print("h2_PT_hd4 = ", h2_PT_hd3.size())
            print("h3_PT_hd4 = ", h3_Cat_hd3.size())
            print("h4_Cat_hd4 = ", hd4_UT_hd3.size())
            print("hd5_UT_hd4 = ", hd5_UT_hd3.size())


        '''create X_DE^2'''
        h1_PT_hd2 = self.h1_PT_de2_Conv_BN_ReLU(self.h1_PT_hd2(x1))
        p2d = (1, 1, 1, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        h1_PT_hd2 = F.pad(h1_PT_hd2, p2d, "constant", 0)
        print("h1_PT_hd2 = ", h1_PT_hd2.size())

        h2_Cat_hd2 = self.h2_PT_de2_Conv_BN_ReLU (x2)
        p2d = (1, 1, 1, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        h2_Cat_hd2 = F.pad(h2_Cat_hd2, p2d, "constant", 0)
        print("h2_Cat_hd2 = ", h2_Cat_hd2.size())

        hd3_UT_hd2 = self.hd3_PT_de2_Conv_BN_ReLU(self.hd3_UT_hd2(hde3))
        print("hd3_UT_hd2 = ", hd3_UT_hd2.size())

        hd4_UT_hd2 = self.hd4_PT_de2_Conv_BN_ReLU(self.hd4_UT_hd2(hde4))
        print("hd4_UT_hd2 = ", hd4_UT_hd2.size())

        hd5_UT_hd2 = self.hd5_PT_de2_Conv_BN_ReLU(self.hd5_UT_hd2(hde5))
        p2d = (4, 4, 4, 4) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        hd5_UT_hd2 = F.pad(hd5_UT_hd2, p2d, "constant", 0)
        print("hd5_UT_hd2 = ", hd5_UT_hd2.size())

        hde2 = self.de2_Conv_BN_ReLU(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)) # hd2->160*160*UpChannels


        print('create X_DE^1')
        '''create X_DE^1'''
        h1_Cat_hd1 = self.h1_PT_de1_Conv_BN_ReLU(x1)
        p2d = (2, 2, 2, 2)
        h1_Cat_hd1 = F.pad(h1_Cat_hd1, p2d, "constant", 0)
        print("h1_Cat_hd1 = ", h1_Cat_hd1.size())
        hd2_UT_hd1 = self.hd2_PT_de1_Conv_BN_ReLU(self.hd2_UT_hd1(hde2))        # output from decode X_DE^2
        print("hd2_UT_hd1 = ", hd2_UT_hd1.size())
        hd3_UT_hd1 = self.hd3_PT_de1_Conv_BN_ReLU(self.hd3_UT_hd1(hde3))
        print("hd3_UT_hd1 = ", hd3_UT_hd1.size())
        hd4_UT_hd1 = self.hd4_PT_de1_Conv_BN_ReLU(self.hd4_UT_hd1(hde4))
        print("hd4_UT_hd1 = ", hd4_UT_hd1.size())
        hd5_UT_hd1 = self.hd5_PT_de1_Conv_BN_ReLU(self.hd5_UT_hd1(hde5))
        p2d = (8, 8, 8, 8)
        hd5_UT_hd1 = F.pad(hd5_UT_hd1, p2d, "constant", 0)
        print("hd5_UT_hd1 = ", hd5_UT_hd1.size())
        hde1 = self.de1_Conv_BN_ReLU(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)) # hd1->320*320*UpChannels

        d1 = self.outconv1(hde1)  # d1->320*320*n_classes
        return torch.sigmoid(d1)
