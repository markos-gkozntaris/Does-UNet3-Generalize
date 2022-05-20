from datetime import datetime
from os import path, makedirs
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm

from data import CTDataset
from model.vanilla_unet_model import UNet


NOW_STR = datetime.now().strftime("%y%m%d%H%M%S")
THIS_DIR = path.dirname(path.abspath(__file__))
DATA_DEFAULT_DIR = path.join(THIS_DIR, '../data/LiverCT')
SAVE_DIR = path.join(THIS_DIR, '../checkpoints')
SAVE_PATH = path.join(SAVE_DIR, NOW_STR + '.pth')
if not path.exists(SAVE_DIR):
    makedirs(SAVE_DIR)

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--data', type=str, default=DATA_DEFAULT_DIR, help='path of the dir containg the data')
parser.add_argument('--train-batch-size', type=int, default=1, help='batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1, help='batch size for testing')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
dataset = CTDataset(args.data, window_size=3)
# split 80-20
train_size = int(0.8 * len(dataset))
train_dataset = Subset(dataset, torch.arange(train_size))
test_dataset = Subset(dataset, torch.arange(train_size, len(dataset)))
# create loaders
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

# define net, optimizer and criterion
net = UNet(n_channels=3, n_classes=1, pad='pad')
net.to(device=device)
optimizer = Adam(net.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss()


def train(train_loader, net, optimizer, criterion):
    sum_loss = 0
    for i, (ct_slices, masks) in enumerate(train_loader):  # ct_slices: (1,3,256,256), masks: (1,3,248,248)
        print(i, '/', len(train_loader)) # BUG at iteration 71 : RuntimeError: Given groups=1, weight of size [64, 3, 3, 3], expected input[1, 0, 256, 256] to have 3 channels, but got 0 channels instead
        optimizer.zero_grad()
        segmentations = net(ct_slices)
        loss = criterion(segmentations, masks)
        loss.backward()
        optimizer.step()
        sum_loss += loss
    return sum_loss / len(train_loader)


def test(test_loader, net, criterion):
    sum_loss = 0
    with torch.no_grad():
        for ct_slices, masks in test_loader:
            segmentations = net(ct_slices)
            loss = criterion(segmentations, masks)
            sum_loss += loss
    return sum_loss / len(test_loader)


train_losses = []
test_losses = []
for epoch in tqdm(range(args.epochs)):
    avg_train_loss = train(train_loader, net, optimizer, criterion)
    avg_test_loss = test(test_loader, net, criterion)
    train_losses.append(avg_train_loss.item())
    test_losses.append(avg_test_loss.item())

torch.save(net.state_dict(), SAVE_PATH)

print('train losses: ', train_losses)
print('test losses: ', test_losses)
