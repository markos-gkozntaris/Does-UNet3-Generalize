from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm

from data import CTDataset
from model.vanilla_unet_model import UNet


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--data', type=str, default='../data/LiverCT/', help='path of the dir containg the data')
parser.add_argument('--slices', type=str, default='../data/liver_n_slices.csv', help='path to the csv with numbers of slices of each CT')
parser.add_argument('--train-batch-size', type=int, default=1, help='batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1, help='batch size for testing')
args = parser.parse_args()

# load data
dataset = CTDataset(args.data, args.slices, window_size=3)
# split 80-20
train_size = int(0.8 * len(dataset))
train_dataset = Subset(dataset, torch.arange(20))# train_dataset = Subset(dataset, torch.arange(train_size))
test_dataset = Subset(dataset, torch.arange(20, 25))# test_dataset = Subset(dataset, torch.arange(train_size, len(dataset)))
# create loaders
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)#train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=0) # TODO increase workers and see how it goes
test_loader = DataLoader(test_dataset, batch_size=1)#test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

# define net, optimizer and criterion
net = UNet(n_channels=3, n_classes=2, pad='pad')
optimizer = Adam(net.parameters(), lr=5e-4)
criterion = nn.BCELoss()


def train(train_loader, net, optimizer, criterion):
    sum_loss = 0
    for i, (ct_slices, masks) in enumerate(train_loader):  # ct_slices: (1,3,256,256) uint8, masks: (1,3,256,256) uint8
        # FIXME: probably needs floats instead of uint8 afterall
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
            loss = criterion(segmentations)
            sum_loss += loss
    return sum_loss / len(test_loader)


train_losses = []
test_losses = []
for epoch in tqdm(range(args.epochs)):
    avg_train_loss = train(train_loader, net, optimizer, criterion)
    avg_test_loss = test(test_loader, net, criterion)
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

print(train_losses)
print(test_losses)
