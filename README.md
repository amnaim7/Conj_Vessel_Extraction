# Dilated Attention U-Net for Conjunctival Vessel Segmentation

This repository contains a PyTorch implementation of a **Dilated Attention U-Net** designed for the segmentation of **diameter-specific conjunctival blood vessels** from high-resolution ocular images.

The model integrates **dilated convolutions**, **multi-head self- and cross-attention**, and a **U-Net encoder–decoder architecture** to capture both fine vessel structures and long-range contextual information. The pipeline supports training on resized images and **patch-based inference with overlap-aware reassembly** for full-resolution evaluation.

---

## ✨ Key Features

- U-Net–style encoder–decoder architecture  
- Dilated convolution bottleneck for enlarged receptive fields  
- Multi-head self-attention (MHSA) at the bottleneck  
- Multi-head cross-attention (MHCA) for skip-connection fusion  
- Patch-based inference for very large images  
- Overlap-aware reconstruction via probability averaging  
- Binary segmentation metrics (Dice, IoU, Accuracy, MCC, etc.)  
- Clean, modular PyTorch codebase  

---

## 🧠 Model Architecture

The proposed architecture extends a standard U-Net with:

- Encoder–decoder convolutional blocks  
- Dilated convolution block (D-Block) to enhance multi-scale context  
- Multi-head self-attention applied at the bottleneck  
- Multi-head cross-attention between encoder features and decoder queries  
- Positional encoding to preserve spatial information  

This design improves vessel continuity and sensitivity to vessels of varying diameters.

---

## 📁 Repository Structure

```
src/
├── data/
│   ├── Dataset.py
│   └── get_loaders.py
├── models/
│   ├── DilTransAttUNet.py
│   └── EncoderDecoder.py
├── inference/
│   └── patch_inference.py
├── utils/
│   ├── Metrics.py
│   ├── losses.py
│   ├── model_analysis.py
│   └── config.py
├── solver.py
```

---

## 🧪 Data Format

The code expects **TIFF images** organized as:

```
data/
├── Patient_001/
│   ├── Patient_001_1.tif
│   ├── Patient_001_1_mask.tif
│   └── ...
```

- Masks use the suffix `_mask.tif`
- Samples with empty masks are automatically filtered

---

## 🚀 Training

```python
train_loader, val_loader = get_loaders(
    data_dir="path/to/data",
    train_ratio=0.8,
    batch_size=4,
)

solver.fit()
```

---

## 🧩 Patch-Based Inference

Large images are processed using overlapping patches:

- Patch size: 512 × 512  
- Stride: 482  
- Overlapping predictions are averaged before thresholding  

---

## 📦 Requirements

```
ipykernel==6.17.1
jupyter_core==5.9.1
matplotlib==3.10.0
numpy==2.0.2
opencv-python==4.13.0.92
scipy==1.16.3
scikit-learn==1.6.1
sklearn-pandas==2.2.0
torch==2.9.0+cu128
torchaudio==2.9.0+cu128
torchmetrics==1.8.2
torchsummary==1.5.1
torchvision==0.24.0+cu128
tqdm==4.67.3
```

---

## 📄 License

This repository is intended for research and academic use.
