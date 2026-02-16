import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch

try:
    from thop import profile  # pip install thop
except Exception:
    profile = None


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_macs_thop(
    model: torch.nn.Module,
    device: torch.device,
    input_size=(1, 3, 256, 256),
) -> Tuple[Optional[float], Optional[int]]:
    """
    Computes MACs and params using thop (if available).
    Returns (macs, params). If thop isn't installed, returns (None, None).
    """
    if profile is None:
        return None, None

    model.eval()
    dummy = torch.randn(*input_size, device=device)

    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy,), verbose=False)

    return float(macs), int(params)


def measure_inference_time(
    model: torch.nn.Module,
    device: torch.device,
    input_size=(1, 3, 256, 256),
    warmup_runs=10,
    timed_runs=100,
) -> float:
    """Average inference time per forward pass in ms."""
    model.eval()
    x = torch.randn(*input_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(timed_runs):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    return (end - start) / timed_runs * 1000.0


def measure_peak_memory(
    model: torch.nn.Module,
    device: torch.device,
    input_size=(1, 3, 256, 256),
) -> Optional[float]:
    """Peak GPU memory usage during inference in MB. Returns None on CPU."""
    if device.type != "cuda":
        return None

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(*input_size, device=device)

    with torch.no_grad():
        _ = model(x)

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return float(peak_mb)


def measure_model_size_mb(model: torch.nn.Module, filename="temp_model.pth") -> float:
    """Serialized state_dict size in MB."""
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024**2)
    os.remove(filename)
    return float(size_mb)


def analyse_model(
    model: torch.nn.Module,
    device: torch.device,
    input_size=(1, 3, 256, 256),
) -> Dict[str, object]:
    """
    Full computational & practical analysis for a PyTorch model.
    """
    model = model.to(device).eval()

    total_params, trainable_params = count_parameters(model)

    macs, thop_params = compute_macs_thop(model, device, input_size=input_size)
    flops = (macs * 2) if macs is not None else None  # common convention

    inference_ms = measure_inference_time(model, device, input_size=input_size)
    peak_mem_mb = measure_peak_memory(model, device, input_size=input_size)
    size_mb = measure_model_size_mb(model)

    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "THOP Params": thop_params,             # can differ from state params w/ wrappers
        "MACs (G)": None if macs is None else macs / 1e9,
        "FLOPs (G)": None if flops is None else flops / 1e9,
        "Inference Time (ms)": inference_ms,
        "Peak GPU Memory (MB)": peak_mem_mb,
        "Model Size (MB)": size_mb,
        "Input Size": input_size,
        "Device": str(device),
    }
