import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib


DICOM_LIVER = (54, 66)


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
            # load CT
            loaded_ct = load_nii(os.path.join(self.data_path, f'volumes/volume-{ct_n}.nii'))
            # apply DICOM filter to CT
            loaded_ct = torch.clip(loaded_ct, DICOM_LIVER[0], DICOM_LIVER[1])
            # scale CT to [0, 1]
            loaded_ct = (loaded_ct - DICOM_LIVER[0]) / (DICOM_LIVER[1] - DICOM_LIVER[0])
            # resample to 256x256 via max pooling
            loaded_ct = F.max_pool2d(loaded_ct, kernel_size=2, stride=2)
            # convert to uint8
            loaded_ct = (loaded_ct * 255).to(torch.uint8)
            # cache
            self.cached_ct = loaded_ct

            # load mask
            loaded_mask = load_nii(os.path.join(self.data_path, f'segmentations/segmentation-{ct_n}.nii'))
            # replace 2's with 1's
            loaded_mask[loaded_mask > 1] = 1
            # resample to 256x256 via max pooling
            loaded_mask = F.max_pool2d(loaded_mask, kernel_size=2, stride=2)
            # convert to uint8
            loaded_mask = loaded_mask.to(torch.uint8)
            # cache
            self.cached_mask = loaded_mask

        return self.cached_ct, self.cached_mask

    def __getitem__(self, idx):
        ct_n = torch.sum(self.win_ct_cummul <= idx).item()
        slice_n = idx - self.win_ct_cummul[ct_n]
        # load nth ct and mask
        ct, mask = self.load_ct_and_mask(ct_n)
        # get window_size consecutive slices to use as channels
        lower_idx = slice_n - (self.window_size // 2)
        upper_idx = slice_n + (self.window_size // 2) + 1  # + 1 because [,) range in slices
        ct_window = ct[lower_idx:upper_idx, ...]
        mask_window = mask[lower_idx:upper_idx, ...]
        return ct_window, mask_window
