import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


@dataclass
class Pair:
    image_path: str
    mask_path: str


def list_image_mask_pairs(data_dir: str) -> List[Pair]:
    """
    Lists (image, mask) pairs under a single data root.

    Expected structure:
      data_dir/
        Patient_001/
          Patient_001_1.tif
          Patient_001_1_mask.tif
          ...
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"{data_dir} is not a valid directory")

    pairs: List[Pair] = []

    for patient in sorted(os.listdir(data_dir)):
        pdir = os.path.join(data_dir, patient)
        if not os.path.isdir(pdir):
            continue

        for fname in sorted(os.listdir(pdir)):
            if not fname.endswith("_mask.tif"):
                continue

            mask_path = os.path.join(pdir, fname)
            image_path = mask_path.replace("_mask.tif", ".tif")

            if os.path.exists(image_path):
                pairs.append(Pair(image_path=image_path, mask_path=mask_path))

    return pairs


def preprocess_patch_rgb(patch: Image.Image) -> torch.Tensor:
    """
    patch: PIL RGB image
    returns: float tensor [1, 3, H, W] in [0, 1]
    """
    arr = np.array(patch).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


@torch.inference_mode()
def predict_full_image_patchwise(
    model: torch.nn.Module,
    pil_img: Image.Image,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 482,
) -> torch.Tensor:
    """
    Patchwise inference with overlap reassembly by averaging probabilities in overlap regions.

    Returns:
      prob_map: torch.Tensor [H, W] on device (float in [0,1])
    """
    model.eval()

    W, H = pil_img.size

    full_probs = torch.zeros((1, 1, H, W), device=device)
    count_map = torch.zeros((1, 1, H, W), device=device)

    # Slide window
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + patch_size, H)
            right = min(left + patch_size, W)

            patch = pil_img.crop((left, top, right, bottom))
            x = preprocess_patch_rgb(patch).to(device)

            # pad to (patch_size, patch_size) if we're at borders
            _, _, ph, pw = x.shape
            if ph < patch_size or pw < patch_size:
                pad_h = patch_size - ph
                pad_w = patch_size - pw
                x = F.pad(x, (0, pad_w, 0, pad_h))

            logits = model(x)                 # [1, 1, patch, patch] (expected)
            probs = torch.sigmoid(logits)     # probabilities

            # unpad
            probs = probs[:, :, : (bottom - top), : (right - left)]

            full_probs[:, :, top:bottom, left:right] += probs
            count_map[:, :, top:bottom, left:right] += 1.0

    avg_probs = full_probs / torch.clamp(count_map, min=1.0)
    return avg_probs.squeeze(0).squeeze(0)  # [H, W]


def load_mask_as_tensor(mask_path: str, device: torch.device) -> torch.Tensor:
    """
    Loads a mask file and returns float tensor [H, W] in {0,1}.
    """
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    gt = (gt > 0).astype(np.float32)  # your masks are tif; >0 matches your earlier filtering
    return torch.from_numpy(gt).to(device)


def save_binary_mask(pred_bin_01: np.ndarray, save_path: str) -> None:
    """
    pred_bin_01: uint8 array {0,1} shape [H,W]
    Saves as 0/255 PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = (pred_bin_01 * 255).astype(np.uint8)
    cv2.imwrite(save_path, out)


@torch.inference_mode()
def run_patchwise_test(
    model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    metrics,  # your existing Metrics class instance
    save_dir: Optional[str] = None,
    patch_size: int = 512,
    stride: int = 482,
    threshold: float = 0.5,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """
    Runs patchwise inference across all pairs in data_dir and prints/returns final metrics.

    metrics: should be your existing Metrics wrapper with reset/update/compute.
             It should accept (logits or preds?) depending on your implementation.

    IMPORTANT:
      We update metrics using *binary predictions* (0/1) since our Metrics wrapper
      typically expects logits and thresholds internally. But for full-image inference,
      it's often cleaner to pass *logits-like* or pass probabilities.

      To avoid mismatch, we pass "logits equivalent" by using logit(prob) if you want.
      Instead, easiest: change Metrics wrapper to accept preds already.
      But since you said you already use it, we will pass *logits* by converting probs to logits.
    """
    pairs = list_image_mask_pairs(data_dir)
    if limit is not None:
        pairs = pairs[:limit]

    if len(pairs) == 0:
        raise ValueError(f"No image/mask pairs found under {data_dir}")

    # reset metric accumulator
    metrics.reset()

    for pair in tqdm(pairs, desc="Patchwise Test"):
        pil_img = Image.open(pair.image_path).convert("RGB")

        prob_map = predict_full_image_patchwise(
            model=model,
            pil_img=pil_img,
            device=device,
            patch_size=patch_size,
            stride=stride,
        )  # [H,W] float in [0,1]

        # Average probs already done; now threshold to get binary mask
        pred_bin = (prob_map >= threshold).to(torch.int32)  # [H,W] int

        # Save if requested
        if save_dir is not None:
            # keep file name, store as png
            base = os.path.splitext(os.path.basename(pair.image_path))[0]
            save_path = os.path.join(save_dir, f"{base}_pred.png")
            save_binary_mask(pred_bin.cpu().numpy().astype(np.uint8), save_path)

        gt = load_mask_as_tensor(pair.mask_path, device=device).to(torch.int32)  # [H,W]

        # Your Metrics.update likely expects (logits, target) and thresholds internally.
        # Here we already have binary preds. We can pass preds directly if Metrics is written for preds.
        # If your Metrics class expects logits, simplest is to fake logits from binary preds:
        # logits = +inf for 1 and -inf for 0 is not numerically stable, so use +/- 10.
        logits_like = pred_bin.float() * 20.0 - 10.0  # 1->+10, 0->-10

        # Add batch dims: [1,1,H,W]
        logits_like = logits_like.unsqueeze(0).unsqueeze(0)
        gt = gt.unsqueeze(0).unsqueeze(0)

        metrics.update(logits_like, gt)

    final = metrics.compute()

    # Convert tensors to floats
    out = {k: float(v.detach().cpu().item()) for k, v in final.items()}
    return out
