from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm

from data import CTDataset
from model.unet3 import UNet3Plus
from util import get_device


EPOCHS=1
DATA='../data/LiverCT'
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1

device = get_device()

# load data
dataset = CTDataset(DATA, window_size=3)
# split 80-20
train_size = int(0.8 * len(dataset))
train_dataset = Subset(dataset, torch.arange(train_size))
test_dataset = Subset(dataset, torch.arange(train_size, len(dataset)))
# create loaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

# define net, optimizer and criterion
net = UNet3Plus(n_channels=3, n_classes=1, print=True)
net.to(device=device)
optimizer = Adam(net.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss()


def train(train_loader, net, optimizer, criterion):
    sum_loss = 0
    for i, (ct_slices, masks) in enumerate(train_loader):  # ct_slices: (1,3,256,256), masks: (1,3,248,248)
        if i % 10 == 0:
            print(i, '/', len(train_loader))
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
for epoch in tqdm(range(EPOCHS)):
    avg_train_loss = train(train_loader, net, optimizer, criterion)
    avg_test_loss = test(test_loader, net, criterion)
    train_losses.append(avg_train_loss.item())
    test_losses.append(avg_test_loss.item())

NOW_STR = datetime.now().strftime("%d%H%M")
torch.save(net.state_dict(), f'{NOW_STR}-{TRAIN_BATCH_SIZE}-{EPOCHS}.pth')

print('train losses: ', train_losses)
print('test losses: ', test_losses)
