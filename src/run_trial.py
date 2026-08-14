"""Run ONE inference trial and emit its measurements as a single JSON line.

This file must remain byte-identical across all systems; the only thing that
varies between systems is the --os-label argument. No network calls are made:
the model weights and test set are loaded from local files produced once by
model_setup.py and copied to every machine.

Measured per trial (over the fixed test set):
    accuracy, macro F1            -> "performance"
    total wall time, ms/image     -> "processing time"
    peak RSS (MB), CPU percent    -> "expense" substitute (physical machines)

Output: a line starting with RESULT_JSON: followed by a JSON object.
run_experiment.py parses this; the script can also be run standalone.
"""

import argparse
import json
import platform
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

SEED = 42
NUM_CLASSES = 10
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PeakMemorySampler:
    """Samples this process's RSS in a background thread and records the peak.

    psutil exposes peak memory natively only on Windows (peak_wset); sampling
    gives one identical cross-platform mechanism on all four systems.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.proc = psutil.Process()
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakMemorySampler":
        self.peak_rss = self.proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        rss = self.proc.memory_info().rss
        if rss > self.peak_rss:
            self.peak_rss = rss


def preprocess(batch_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW 32x32 -> normalized float32 NCHW 224x224 (part of timed work)."""
    x = batch_uint8.float().div_(255.0)
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os-label", required=True,
                        help="ubuntu | win11_machine1 | fedora | win11_machine2")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4,
                        help="fixed torch thread count; keep identical on all systems")
    parser.add_argument("--warmup-batches", type=int, default=2)
    args = parser.parse_args()

    # Fix every controllable source of nondeterminism so that remaining
    # variability is attributable to the OS/system, not to our code.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")  # CPU-only by design: OS comparison, not driver comparison

    data_dir = Path(args.data_dir)

    # ---- untimed setup: load everything into memory before measuring ----
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(data_dir / "model_weights.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    test = torch.load(data_dir / "test_set.pt", map_location=device, weights_only=True)
    images, labels = test["images"], test["labels"]
    n_images = images.shape[0]
    batches = [images[i:i + args.batch_size] for i in range(0, n_images, args.batch_size)]

    # ---- warmup (untimed): stabilizes allocator caches and CPU frequency ----
    with torch.inference_mode():
        for b in batches[: args.warmup_batches]:
            model(preprocess(b))

    # ---- timed inference over the full test set ----
    proc = psutil.Process()
    proc.cpu_percent(None)  # reset the CPU counter window
    all_preds = []
    with PeakMemorySampler() as mem, torch.inference_mode():
        t0 = time.perf_counter()
        for b in batches:
            out = model(preprocess(b))
            all_preds.append(out.argmax(1))
        total_time_s = time.perf_counter() - t0
    cpu_percent = proc.cpu_percent(None)  # mean over the timed window

    preds = torch.cat(all_preds)
    accuracy = (preds == labels).float().mean().item()

    result = {
        "os_label": args.os_label,
        "run_idx": args.run_idx,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_images": n_images,
        "accuracy": round(accuracy, 6),
        "f1_macro": round(macro_f1(preds, labels, NUM_CLASSES), 6),
        "total_time_s": round(total_time_s, 6),
        "per_image_ms": round(total_time_s * 1000.0 / n_images, 6),
        "peak_rss_mb": round(mem.peak_rss / (1024 * 1024), 3),
        "cpu_percent": round(cpu_percent, 2),
        "batch_size": args.batch_size,
        "num_threads": args.threads,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
