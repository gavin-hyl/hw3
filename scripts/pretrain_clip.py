"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    p.add_argument("--pos-encoding", default="learned",
                   choices=["learned", "rope_1d", "rope_2d"],
                   help="§6 ablation: positional-encoding scheme for the ViT.")
    p.add_argument("--eval-img-size", type=int, default=None,
                   help="§6 length-extrapolation test: evaluate at this image size.")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    from basics.text_encoder import FrozenTextEncoder
    from basics.vit import ViT
    from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
    from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders
    from vlm.eval import zeroshot_classification_accuracy

    tcfg = cfg["train"]
    num_epochs = args.epochs or tcfg["num_epochs"]
    train_dl, val_dl, _ = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"], batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
    )

    vit_cfg = dict(cfg["vit"])
    vit = ViT(**vit_cfg, pos_encoding=args.pos_encoding).to(device)
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"]).to(device)
    heads = ProjectionHeads(vit.d_model, text_encoder.embedding_dim,
                            cfg["projection"]["d_proj"]).to(device)
    logit_scale = init_logit_scale().to(device)

    params = list(vit.parameters()) + list(heads.parameters()) + [logit_scale]
    ocfg = cfg["optim"]
    opt = torch.optim.AdamW(params, lr=ocfg["lr"], weight_decay=ocfg["weight_decay"],
                            betas=tuple(ocfg["betas"]))
    total_steps = num_epochs * len(train_dl)
    sched = cosine_warmup(opt, ocfg["warmup_steps"], total_steps)

    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    history = {"train_loss": [], "val_acc": []}
    best_acc = -1.0
    for epoch in range(1, num_epochs + 1):
        vit.train(); heads.train()
        running, n = 0.0, 0
        for step, (images, captions) in enumerate(train_dl):
            images = images.to(device)
            with torch.no_grad():
                text_embeds = text_encoder(captions).to(device)
            img_feats = vit(images)
            img_proj, txt_proj = heads(img_feats, text_embeds)
            loss = clip_loss(img_proj, txt_proj, logit_scale)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            logit_scale.data.clamp_(max=math.log(100.0))  # CLIP temperature cap
            running += loss.item() * images.size(0); n += images.size(0)
            if step % tcfg["log_every"] == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}")
        train_loss = running / max(n, 1)

        acc = zeroshot_classification_accuracy(
            vit, heads, text_encoder, val_dl, class_prompts, class_indices, device
        )
        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)
        print(f"[epoch {epoch}] train_loss {train_loss:.4f} | zero-shot val acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {"vit": vit.state_dict(), "heads": heads.state_dict(),
                 "logit_scale": logit_scale.detach().cpu(),
                 "vit_config": vit_cfg, "pos_encoding": args.pos_encoding},
                args.output_dir / "best.pt",
            )

    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best zero-shot val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
