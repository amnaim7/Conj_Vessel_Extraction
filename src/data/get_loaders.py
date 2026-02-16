import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from .Dataset import Dataset


def get_loaders(
    data_dir: str,
    train_ratio: float,
    batch_size: int,
    size=(512, 512),
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Returns train/val DataLoaders only.

    Expected structure:
      data_dir/
        Patient_001/
          Patient_001_1.tif
          Patient_001_1_mask.tif
          ...
        Patient_002/
          ...

    Filters out samples where mask is empty (max pixel == 0).
    """
    images, masks = get_image_mask_pairs(data_dir, seed=seed)

    n = len(images)
    if n == 0:
        raise ValueError(f"No valid image/mask pairs found under: {data_dir}")

    train_end = int(n * train_ratio)
    if not (0 < train_end < n):
        raise ValueError(f"Invalid train_ratio={train_ratio} for n={n}")

    train = (images[:train_end], masks[:train_end])
    val = (images[train_end:], masks[train_end:])

    train_ds = Dataset(train[0], train[1], size)
    val_ds = Dataset(val[0], val[1], size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_image_mask_pairs(data_dir: str, seed: int = 42):
    """
    Collect (image, mask) pairs from patient subdirectories.
    Keeps only pairs where mask has at least one non-zero pixel.
    Shuffles in unison.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"{data_dir} is not a valid directory")

    images = []
    masks = []

    for patient in tqdm(sorted(os.listdir(data_dir)), desc="Patients"):
        patient_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_dir):
            continue

        # iterate mask files; derive image path from mask name
        for fname in sorted(os.listdir(patient_dir)):
            if not fname.endswith("_mask.tif"):
                continue

            mask_path = os.path.join(patient_dir, fname)
            image_path = mask_path.replace("_mask.tif", ".tif")
            if not os.path.exists(image_path):
                continue

            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue

            if int(np.max(m)) > 0:
                images.append(image_path)
                masks.append(mask_path)

    # deterministic shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    images = [images[i] for i in idx]
    masks = [masks[i] for i in idx]

    return images, masks
