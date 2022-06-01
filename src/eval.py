from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad

from data import CTDataset
from util import visualize_slice, get_device
from model.vanilla_unet_model import UNet


device = get_device()

parser = ArgumentParser()
parser.add_argument('checkpoint', type=str, help='checkpoint of the model to load')
parser.add_argument('--data', type=str, default='../data/LiverCT', help='path to data dir')
args = parser.parse_args()

dataset = CTDataset(args.data, window_size=3)
model = UNet(n_channels=3, n_classes=1, pad='pad')
model.to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

slice = 55
example_ct, example_mask = dataset[slice]
example_ct = example_ct.unsqueeze(0)
with torch.no_grad():
    pred_mask = model(example_ct)
pred_mask = pad(pred_mask, 4 * (4,))
print(example_ct.shape)
print(pred_mask.shape)
ax = visualize_slice(example_ct[0][1], pred_mask[0][0])
plt.show()
