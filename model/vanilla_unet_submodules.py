import torch
import torch.nn as nn
import torch.nn.functional as F


# vanilla model uses two convolutions before every down or up sampling 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # set padding = 0 for dimensions exactly the same with the original
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),     # currently padding = valid (original paper has no padding)
            # batchnorm probably not on original paper, both 2015 papers so yeah ...
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, padding_type):
        super().__init__()
        self.padding_type = padding_type
        # half of the number of channels comes from concatination
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # up x1 before concatinating
        x1 = self.up(x1)
        if self.padding_type == "pad":
            # input is CHW
            # NOTE: in the original paper they are cropping, while here we are padding so there will be some output mismatch with original paper
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
    
            # pad works as follows for 2d arrays:  [padding_left, padding_right, padding_up, padding_down]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2], "constant", 0)
        else:
        # cropping would be 
            x2 = x2[:,:, (x2.size()[2]-x1.size()[2])//2:x2.size()[2]-(x2.size()[2]-x1.size()[2])//2, (x2.size()[3]-x1.size()[3])//2:x2.size()[3]-(x2.size()[3]-x1.size()[3])//2]

        # basically concatinating and increasing the number of channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)