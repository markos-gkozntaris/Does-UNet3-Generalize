import os

import torch
from torch.utils.data import Dataset
import nibabel as nib


def load_nii(file_path):
    ct_scan = nib.load(file_path).get_fdata()
    tensor = torch.tensor(ct_scan, dtype=torch.float32)
    tensor.transpose_(0, 2)
    return tensor


class CTDataset(Dataset): 
    def __init__(self, data_path, n_slices_path, window_size=3):
        # window_size is the size of the rolling window over all the slices
        self.data_path = data_path
        self.n_slices_path = n_slices_path
        self.window_size = window_size

        self.last_n_cached = -1
        self.cached_ct = None
        self.cached_mask = None

        with open(self.n_slices_path, 'r') as f:
            n_slices_raw = f.readlines()
        win_ct = []
        for line in n_slices_raw:
            n = int(line.split(',')[1])
            n -= 2 * (self.window_size // 2)
            win_ct.append(n)
        self.win_ct = torch.tensor(win_ct)
        self.win_ct_cummul = self.win_ct.cumsum(dim=0)

    def __len__(self):
        return self.win_ct_cummul[-1]

    def load_ct_and_mask(self, ct_n):
        # some caching logic to avoid loading the nii file everytime
        if self.last_n_cached != ct_n:
            self.last_n_cached = ct_n
            self.cached_ct = load_nii(os.path.join(self.data_path, f'volumes/volume-{ct_n}.nii'))
            self.cached_mask = load_nii(os.path.join(self.data_path, f'segmentations/segmentation-{ct_n}.nii'))
        return self.cached_ct, self.cached_mask

    def __getitem__(self, idx):
        ct_n = torch.sum(self.win_ct_cummul <= idx).item()
        slice_n = idx - self.win_ct_cummul[ct_n]
        # load nth ct and mask
        ct, mask = self.load_ct_and_mask(ct_n)
        ct_slice = ct[slice_n, ...]
        mask_slice = mask[slice_n, ...]
        return ct_slice, mask_slice
