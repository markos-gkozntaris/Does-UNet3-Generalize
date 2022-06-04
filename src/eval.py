import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad

from data import CTDataset                    # ANCHOR_REMOVE_LINE_KAGGLE
from util import visualize_slice, get_device  # ANCHOR_REMOVE_LINE_KAGGLE
from model.vanilla_unet_model import UNet     # ANCHOR_REMOVE_LINE_KAGGLE


CHECKPOINT=''
DATA='../data/LiverCT'


device = get_device()

dataset = CTDataset(DATA, window_size=3)
model = UNet(n_channels=3, n_classes=1, pad='pad')
model.to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
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
