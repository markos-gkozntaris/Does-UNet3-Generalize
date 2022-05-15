from argparse import ArgumentParser

from tqdm import tqdm

from data import CTDataset
from util import visualize


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--data', type=str, default='../data/LiverCT/', help='path of the dir containg the data')
parser.add_argument('--slices', type=str, default='../data/liver_n_slices.csv', help='path to the csv with numbers of slices of each CT')
args = parser.parse_args()


dataset = CTDataset(args.data, args.slices)
print(len(dataset))
