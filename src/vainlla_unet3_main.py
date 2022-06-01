import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from model.unet3 import UNet3Plus
from model.unet3_submodules import *


# call UNet with pad = "pad" if you want to pad the image, else it will crop as the original paper does
# TODO: check some mismatch dimensions when cropping

if __name__ == "__main__":
    image = torch.rand((1,3,572,572))
    print(image.size())

    # image = torch.rand((1,64,572,572))
    
    # model1 = TEMPO(n_channels=3, n_classes=2)
    # print(model1(image))

    # model_3PLUS = UNet3Plus(n_channels=3, n_classes=2, bilinear=True, feature_scale=4,
                #  is_deconv=True, is_batchnorm=True)
    model_3PLUS = UNet3Plus(n_channels=3, n_classes=2, print=True)
    print(model_3PLUS(image))
    
    print("Running complete")

    # Total params: 31,036,546 calculated via summary
    # print(summary(model, input_size = (1,572,572))) 