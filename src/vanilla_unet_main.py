import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from model.vanilla_unet_model import UNet

# call UNet with pad = "pad" if you want to pad the image, else it will crop as the original paper does
# TODO: check some mismatch dimensions when cropping

if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = UNet(n_channels=1, n_classes=2, pad = "pad")
    print(model(image))
    
    print("Running complete")

    # Total params: 31,036,546 calculated via summary
    # print(summary(model, input_size = (1,572,572))) 
