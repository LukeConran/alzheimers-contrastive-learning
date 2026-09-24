import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from scipy import ndimage

LABEL_MAP = {
    "Dementia": 0,
    "Mild Cognitive Impairment": 1,
    "Normal Cognition": 2,
}


class MyDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        """
        json_path: path to a JSON file containing {id, label} entries
        image_dir: directory containing all .npz image files (named 0.npz, 1.npz, ...)
        transform: optional image preprocessing function
        """
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        # Keep only samples whose image file actually exists
        self.data = []
        for item in all_data:
            image_path = os.path.join(image_dir, f"{item['id']}.npz")
            if os.path.exists(image_path):
                self.data.append(item)
            else:
                print(f"Skipping missing image: {item['id']}.npz")

        # Sort by id (optional)
        self.data = sorted(self.data, key=lambda x: x["id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{item['id']}.npz")
        image_npz = np.load(image_path)
        image = image_npz["image_mr"]  # shape: (D, H, W)
        image = self.__data_process__(image)
        image_tensor = torch.from_numpy(image).float()

        label_text = item["label"]
        label = LABEL_MAP[label_text]

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label

    def __data_process__(self, data):

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    def __resize_data__(self, data):
        '''
        Resize the data to the input size
        '''

        [channel, depth, height, width] = data.shape
        scale = [channel, 79 * 1.0 / depth, 95 * 1.0 / height, 79 * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __itensity_normalize_one_volume__(self, volume):
        '''
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        '''

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# ContrastiveDataset
# ═══════════════════════════════════════════════════════════════════════════════
# Extends MyDataset to support contrastive learning.
#
# Key difference from MyDataset:
#   __getitem__ returns TWO independently-augmented views of the same MRI scan
#   instead of one. The SupCon loss uses these pairs to learn that same-class
#   scans should cluster together in embedding space.
#
# Augmentation strategy for 3D MRI:
#   We keep augmentations conservative because:
#     (1) Brain anatomy is spatially structured — aggressive crops can remove
#         disease-relevant regions (hippocampus, amygdala, etc.).
#     (2) MRI intensities are already normalized — color jitter is inappropriate.
#   Safe augmentations used here:
#     - Random axis flips (left-right, anterior-posterior, superior-inferior)
#     - Gaussian noise (simulates scanner noise variability)
#     - Intensity scaling (simulates gain/contrast variability across scanners)
# ═══════════════════════════════════════════════════════════════════════════════

class ContrastiveDataset(MyDataset):
    """
    Dataset for supervised contrastive learning.

    Each call to __getitem__ loads one MRI scan and returns two independently
    augmented views (view1, view2) along with the disease class label.

    The two views are fed separately through the encoder, and their embeddings
    are concatenated before being passed to SupConLoss.

    Args:
        json_path (str):    Path to JSON file with {id, label} entries.
        image_dir (str):    Directory containing <id>.npz image files.
        noise_std (float):  Std of Gaussian noise added during augmentation.
                            Default 0.05 adds subtle noise relative to the
                            z-scored volume (mean≈0, std≈1).
        scale_range (tuple): Min/max multiplicative intensity scaling factor.
                             (0.9, 1.1) means ±10% brightness variation.
    """

    def __init__(self, json_path, image_dir, noise_std=0.05, scale_range=(0.9, 1.1)):
        super().__init__(json_path, image_dir)
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{item['id']}.npz")
        image_npz = np.load(image_path)
        image = image_npz["image_mr"]               # (D, H, W), raw volume
        image = self.__data_process__(image)         # intensity normalization (from parent)

        # Apply two DIFFERENT random augmentations to the same normalized volume.
        # Each call to _augment samples new random values, so view1 ≠ view2.
        view1 = self._augment(image)
        view2 = self._augment(image)

        label = LABEL_MAP[item["label"]]

        return (
            torch.from_numpy(view1).float(),    # augmented view 1: (D, H, W)
            torch.from_numpy(view2).float(),    # augmented view 2: (D, H, W)
            label,                              # integer class index
        )

    def _augment(self, volume):
        """
        Apply a random combination of MRI-safe 3D augmentations.

        Each augmentation is applied independently with 50% probability,
        so the two views returned by __getitem__ will almost always differ.

        Args:
            volume (np.ndarray): Normalized 3D volume, shape (D, H, W).

        Returns:
            np.ndarray: Augmented volume, same shape as input.
        """
        volume = volume.copy()   # avoid mutating the shared numpy array

        # ── Augmentation 1: Random axis flips ────────────────────────────────
        # Flipping along each axis is anatomically plausible for MRI
        # (left-right symmetry of the brain is well-established).
        for axis in range(3):
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis).copy()

        # ── Augmentation 2: Additive Gaussian noise ───────────────────────────
        # Simulates thermal/scanner noise. Kept small (noise_std=0.05) relative
        # to the z-scored volume so disease-relevant signal is preserved.
        if np.random.rand() > 0.5:
            volume = volume + np.random.normal(0, self.noise_std, volume.shape)

        # ── Augmentation 3: Intensity scaling ────────────────────────────────
        # Simulates gain variation between MRI scanners or scan sessions.
        # Only applied to non-zero voxels to avoid amplifying background noise.
        if np.random.rand() > 0.5:
            scale = np.random.uniform(*self.scale_range)
            volume[volume != 0] *= scale

        return volume
