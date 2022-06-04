import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib

from util import get_device # ANCHOR_REMOVE_LINE_KAGGLE


DICOM_LIVER = (54, 66)
LIVER_N_SLICES = [75, 123, 517, 534, 841, 537, 518, 541, 541, 549, 501,
466, 455, 605, 588, 565, 689, 826, 845, 547, 574, 437, 247, 391, 276, 601, 668,
861, 129, 172, 200, 91, 139, 135, 151, 124, 111, 122, 132, 260, 122, 113, 125,
155, 119, 74, 124, 225, 244, 254, 240, 227, 237, 105, 96, 192, 239, 366, 212,
216, 244, 193, 188, 104, 230, 513, 86, 165, 266, 245, 333, 94, 93, 121, 107,
89, 168, 94, 198, 147, 217, 343, 519, 871, 733, 630, 647, 896, 811, 766, 751,
751, 836, 696, 917, 841, 722, 671, 645, 629, 685, 683, 677, 683, 781, 986, 771,
771, 856, 756, 816, 761, 751, 836, 846, 846, 908, 836, 427, 461, 424, 463, 422,
432, 407, 410, 401, 987, 654, 338, 624]


device = get_device()


def load_nii(file_path):
    ct_scan = nib.load(file_path).get_fdata()
    tensor = torch.tensor(ct_scan, dtype=torch.float32, device=device)
    tensor.transpose_(0, 2)
    return tensor


class CTDataset(Dataset): 
    def __init__(self, data_path, window_size=3):
        # window_size is the size of the rolling window over all the slices
        self.data_path = data_path
        self.window_size = window_size

        # caches
        self.last_n_cached = -1
        self.cached_ct = None
        self.cached_mask = None

        # files available
        self.volumes_paths = {}
        self.segmentations_paths = {}
        for dirname, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                path = os.path.join(dirname, filename)
                path_num_part = int(path.split('-')[-1].split('.')[0])
                if 'volume' in path:
                    self.volumes_paths[path_num_part] = path
                if 'segmentations' in path:
                    self.segmentations_paths[path_num_part] = path

        # number of windows (batches of slices) available for loading
        n_volumes = len(self.volumes_paths)
        win_ct = [n - (self.window_size - 1) for n in LIVER_N_SLICES[:n_volumes]]
        self.win_ct = torch.tensor(win_ct)
        self.win_ct_cummul = self.win_ct.cumsum(dim=0)

    def __len__(self):
        return self.win_ct_cummul[-1]

    def load_ct_and_mask(self, ct_n):
        # some caching logic to avoid loading the nii file everytime
        if self.last_n_cached != ct_n:
            self.last_n_cached = ct_n
            # load CT
            loaded_ct = load_nii(self.volumes_paths[ct_n])
            # apply DICOM filter to CT
            loaded_ct = torch.clip(loaded_ct, DICOM_LIVER[0], DICOM_LIVER[1])
            # scale CT to [0, 1]
            loaded_ct = (loaded_ct - DICOM_LIVER[0]) / (DICOM_LIVER[1] - DICOM_LIVER[0])
            # resample to 256x256 via max pooling
            loaded_ct = F.max_pool2d(loaded_ct, kernel_size=2, stride=2)
            # cache
            self.cached_ct = loaded_ct

            # load mask
            loaded_mask = load_nii(self.segmentations_paths[ct_n])
            # replace 2's with 1's
            loaded_mask[loaded_mask > 1] = 1
            # resample to 256x256 via max pooling
            loaded_mask = F.max_pool2d(loaded_mask, kernel_size=2, stride=2)
            # cache
            self.cached_mask = loaded_mask

        return self.cached_ct, self.cached_mask

    def __getitem__(self, idx):
        ct_n = torch.sum(self.win_ct_cummul <= idx).item()
        slice_n = idx - self.win_ct_cummul[ct_n - 1] if ct_n != 0 else idx 
        # load nth ct and mask
        ct, mask = self.load_ct_and_mask(ct_n)
        ct_window = ct[slice_n:slice_n + self.window_size, :, :]
        mask_window = mask[slice_n + 1, ...].unsqueeze(dim=0)  # HACK was mask[lower_idx:upper_idx, ...] but UNet cuts 4 pixels from each side so we cut them too from the mask # HACK removed that hack for UNet3+...
        return ct_window, mask_window
