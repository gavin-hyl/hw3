"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class ViTClassifier(nn.Module):
    def __init__(self, vit: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))  # CLS embedding -> class logits


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    from basics.lora import apply_lora_to_attention
    from basics.vit import ViT
    from vlm.data import build_resisc45_loaders

    ckpt = torch.load(args.pretrained, map_location="cpu")
    vit = ViT(**ckpt["vit_config"], pos_encoding=ckpt.get("pos_encoding", "learned"))
    vit.load_state_dict(ckpt["vit"])

    # Apply the adaptation strategy.
    if args.method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad_(False)
    elif args.method == "lora":
        apply_lora_to_attention(vit, args.rank, args.alpha)  # freezes base, adds LoRA
    elif args.method == "full_ft":
        pass  # everything trainable

    model = ViTClassifier(vit, cfg["num_classes"]).to(device)
    # The classification head is always trainable.
    for p in model.head.parameters():
        p.requires_grad_(True)

    lr = cfg["methods"].get(args.method, {}).get("lr", cfg["optim"]["lr"])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=cfg["optim"]["weight_decay"], betas=tuple(cfg["optim"]["betas"]),
    )

    train_dl, test_dl = build_resisc45_loaders(
        batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"]
    )
    num_epochs = args.epochs or cfg["train"]["num_epochs"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        for step, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)
            loss = nn.functional.cross_entropy(model(images), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % cfg["train"]["log_every"] == 0:
                print(f"[{args.method}] epoch {epoch} step {step} loss {loss.item():.4f}")
        acc = evaluate(model, test_dl, device)
        print(f"[{args.method}] epoch {epoch} test_acc {acc:.4f}")
    wall = time.time() - t0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    )
    metrics = {
        "method": args.method,
        "rank": args.rank if args.method == "lora" else None,
        "test_accuracy": evaluate(model, test_dl, device),
        "trainable_params": count_trainable(model),
        "total_params": sum(p.numel() for p in model.parameters()),
        "peak_memory_mb": peak_mem_mb,
        "wall_clock_s": wall,
        "lr": lr,
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
