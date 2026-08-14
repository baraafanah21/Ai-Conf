"""One-time model + data preparation.

Run this ONCE (on the reference machine, Ubuntu) and then copy the resulting
`data/` directory to every other machine/OS partition. This guarantees that
every system runs inference with byte-identical weights and a byte-identical
test set — fine-tuning separately per OS would confound the comparison.

Produces:
    data/model_weights.pt   fine-tuned ResNet18 state_dict (CPU tensors)
    data/test_set.pt        fixed CIFAR-10 test subset (uint8 images + labels)
    data/setup_manifest.json versions + hashes for provenance

Network access happens ONLY here (torchvision downloads). run_trial.py makes
no network calls.
"""

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

SEED = 42
NUM_CLASSES = 10


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="output directory")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seeds()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 download (train + test). Raw images stay uint8 32x32.
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_full = torchvision.datasets.CIFAR10(
        root=str(data_dir / "cifar10"), train=True, download=True, transform=train_tf
    )
    test_full = torchvision.datasets.CIFAR10(
        root=str(data_dir / "cifar10"), train=False, download=True
    )

    # Deterministic subsets: first N indices, no shuffling of the selection.
    train_subset = Subset(train_full, list(range(args.train_size)))
    test_indices = list(range(args.test_size))

    # Save the fixed test subset as raw uint8 tensors; run_trial.py applies the
    # (identical) preprocessing itself so the whole inference pipeline is timed.
    test_images = torch.from_numpy(test_full.data[test_indices]).permute(0, 3, 1, 2).contiguous()
    test_labels = torch.tensor([test_full.targets[i] for i in test_indices], dtype=torch.long)
    torch.save({"images": test_images, "labels": test_labels}, data_dir / "test_set.pt")
    print(f"Saved test set: {tuple(test_images.shape)} uint8 -> {data_dir / 'test_set.pt'}")

    # Light fine-tune: freeze the backbone, train only the new fc head.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on {device} (backbone frozen, fc head only)")
    model = build_model()
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
    model.to(device).train()

    loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # keep single-process for determinism
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={loss_sum / total:.4f}  train_acc={correct / total:.4f}")

    model.eval().cpu()
    weights_path = data_dir / "model_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights -> {weights_path}")

    manifest = {
        "seed": SEED,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "setup_platform": platform.platform(),
        "model_weights_sha256": file_sha256(weights_path),
        "test_set_sha256": file_sha256(data_dir / "test_set.pt"),
    }
    with open(data_dir / "setup_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest -> {data_dir / 'setup_manifest.json'}")
    print(
        "\nNEXT STEP: copy the entire data/ directory (model_weights.pt, test_set.pt, "
        "setup_manifest.json) to every other machine/OS. Verify the SHA-256 hashes match "
        "on each system before running experiments."
    )


if __name__ == "__main__":
    sys.exit(main())
