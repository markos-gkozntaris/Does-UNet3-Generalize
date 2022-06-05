import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad

from data import CTDataset                    # ANCHOR_REMOVE_LINE_KAGGLE
from util import visualize_slice, get_device  # ANCHOR_REMOVE_LINE_KAGGLE
from model.unet3 import UNet3Plus             # ANCHOR_REMOVE_LINE_KAGGLE

CHECKPOINT='../results/bs16-vol0-130-e4/bs16-vol0-130-e4.pth'
DATA='../data/LiverCT'


device = get_device()

dataset = CTDataset(DATA, window_size=3)
model = UNet3Plus(n_channels=3, n_classes=1)
model.to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()


def vis(slice):
    # inference
    real_ct, real_mask = dataset[slice]
    real_ct = real_ct.unsqueeze(0)
    with torch.no_grad():
        pred_mask = model(real_ct)

    #pred_mask = pad(pred_mask, 4 * (4,)) # HACK for UNet (not UNet3)
    real_ct = real_ct.cpu()
    real_mask = real_mask.cpu()
    pred_mask = pred_mask.cpu()

    print(real_ct.shape)
    print(real_mask.shape)
    print(pred_mask.shape)

    # remove extra dimensions
    real_ct = real_ct[0][1]
    real_mask = real_mask[0]
    pred_mask = pred_mask[0][0]

    # plot
    cmap=plt.cm.bone
    fig, ax = plt.subplots(1, 5, figsize=(10, 10))
    ax[0].imshow(real_ct, cmap=cmap)
    ax[0].set_title('ct')
    ax[0].axis('off')

    ax[1].imshow(real_mask, cmap=cmap)
    ax[1].set_title('real mask')
    ax[1].axis('off')

    ax[2].imshow(pred_mask, cmap=cmap)
    ax[2].set_title('pred mask')
    ax[2].axis('off')

    ax[3].imshow(real_ct, cmap=cmap)
    ax[3].imshow(real_mask, alpha=0.5, cmap=cmap)
    ax[3].set_title('ct + real mask')
    ax[3].axis('off')

    ax[4].imshow(real_ct, cmap=cmap)
    ax[4].imshow(pred_mask, alpha=0.5, cmap=cmap)
    ax[4].set_title('ct + pred mask')
    ax[4].axis('off')
    
    return ax


for i in range(0, 1000, 50):
    vis(i)
