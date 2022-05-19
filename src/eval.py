from argparse import ArgumentParser
from os import path

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad

from data import CTDataset
from util import visualize_slice
from model.vanilla_unet_model import UNet


THIS_DIR = path.dirname(path.abspath(__file__))
DATA_DEFAULT_DIR = path.join(THIS_DIR, '../data/LiverCT')
SLICES_PATH = path.join(THIS_DIR, 'resources/liver_n_slices.csv')

parser = ArgumentParser()
parser.add_argument('checkpoint', type=str, help='checkpoint of the model to load')
parser.add_argument('--data', type=str, default=DATA_DEFAULT_DIR, help='path to data dir')
args = parser.parse_args()

dataset = CTDataset(args.data, SLICES_PATH, window_size=3)
model = UNet(n_channels=3, n_classes=1, pad='pad')
model.load_state_dict(torch.load(args.checkpoint))
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
