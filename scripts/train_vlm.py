"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def wrap_decoder_lora(decoder: nn.Module, rank: int, alpha: float) -> nn.Module:
    """Wrap the decoder's q_proj/v_proj linear layers with LoRALinear (§5.6 B)."""
    from basics.lora import LoRALinear

    for p in decoder.parameters():
        p.requires_grad_(False)
    for module in decoder.modules():
        for attr in ("q_proj", "v_proj"):
            child = getattr(module, attr, None)
            if isinstance(child, nn.Linear):
                setattr(module, attr, LoRALinear(child, rank, alpha))
    return decoder


def apply_freeze_config(vlm, freeze_config: str, lora_rank: int, lora_alpha: float) -> None:
    """A: projector only · B: +decoder LoRA · C: +full decoder · D: all three."""
    for p in vlm.vit.parameters():
        p.requires_grad_(freeze_config == "D")
    for p in vlm.projector.parameters():
        p.requires_grad_(True)  # projector is always trained
    if freeze_config in ("C", "D"):
        for p in vlm.decoder.parameters():
            p.requires_grad_(True)
    elif freeze_config == "B":
        wrap_decoder_lora(vlm.decoder, lora_rank, lora_alpha)
    else:  # A
        for p in vlm.decoder.parameters():
            p.requires_grad_(False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from basics.vit import ViT
    from vlm.data import build_clevr_loaders
    from vlm.eval import batch_clevr_accuracy
    from vlm.model import VisionLanguageModel
    from vlm.projector import VisionLanguageProjector
    from vlm.textbatch import IMAGE_TOKEN, build_qa_batch

    # --- ViT (CLIP-pretrained) ---
    ckpt = torch.load(args.pretrained_vit, map_location="cpu")
    vit = ViT(**ckpt["vit_config"], pos_encoding=ckpt.get("pos_encoding", "learned"))
    vit.load_state_dict(ckpt["vit"])

    # --- Decoder + tokenizer ---
    dcfg = cfg["decoder"]
    dtype = getattr(torch, dcfg.get("torch_dtype", "bfloat16"))
    try:
        decoder = AutoModelForCausalLM.from_pretrained(
            dcfg["model_name"], torch_dtype=dtype,
            attn_implementation=dcfg.get("attn_implementation", "flash_attention_2"),
        )
    except Exception as e:  # FlashAttention-2 unavailable -> fall back to SDPA
        print(f"FA2 unavailable ({e}); falling back to sdpa.")
        decoder = AutoModelForCausalLM.from_pretrained(
            dcfg["model_name"], torch_dtype=dtype, attn_implementation="sdpa"
        )
    tokenizer = AutoTokenizer.from_pretrained(dcfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id = None
    if args.injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        decoder.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    projector = VisionLanguageProjector(
        vit.d_model, decoder.config.hidden_size, cfg["projector"]["expansion"]
    )
    vlm = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id)
    apply_freeze_config(vlm, args.freeze_config, lora_rank=8, lora_alpha=16.0)
    # Unify decoder dtype (LoRA adapters are added in fp32 — cast to the decoder
    # dtype so matmuls don't mix dtypes), keep ViT+projector in fp32 for stability.
    vlm.decoder.to(dtype=dtype)
    vlm.to(device)
    vlm.projector.float()

    # --- Optimizer ---
    ocfg, tcfg = cfg["optim"], cfg["train"]
    trainable = [p for p in vlm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=ocfg["lr"], weight_decay=ocfg["weight_decay"],
                            betas=tuple(ocfg["betas"]))
    num_steps = tcfg["num_steps"]
    accum = tcfg.get("gradient_accumulation_steps", 1)

    def lr_lambda(step):
        w = ocfg["warmup_steps"]
        if step < w:
            return (step + 1) / max(1, w)
        prog = (step - w) / max(1, num_steps - w)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_dl, val_dl = build_clevr_loaders(
        batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"]
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_iter = itertools.cycle(train_dl)
    best_acc, history = -1.0, {"loss": [], "val_acc": []}
    t0 = time.time()
    opt.zero_grad()
    for step in range(1, num_steps + 1):
        vlm.train()
        batch = next(train_iter)
        images = batch["image"].to(device)
        input_ids, attn, labels = build_qa_batch(
            tokenizer, batch["question"], batch["answer"], args.injection, device
        )
        out = vlm(images, input_ids, attn, labels=labels,
                  injection=args.injection, mask_mode=args.mask_mode)
        loss = out["loss"] / accum
        loss.backward()
        if step % accum == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if step % tcfg["log_every"] == 0:
            print(f"step {step}/{num_steps} loss {out['loss'].item():.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}")
            history["loss"].append(out["loss"].item())

        if step % tcfg["eval_every_steps"] == 0 or step == num_steps:
            acc = run_eval(vlm, val_dl, args.injection, device, tcfg["eval_max_examples"],
                           cfg["generation"]["max_new_tokens"])
            history["val_acc"].append({"step": step, "acc": acc})
            print(f"[eval @ {step}] val exact-match acc {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(args, vlm, ckpt["vit_config"], ckpt.get("pos_encoding"),
                                dcfg["model_name"], image_token_id)

    peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    summary = {"best_val_acc": best_acc, "peak_memory_mb": peak,
               "wall_clock_s": time.time() - t0,
               "trainable_params": sum(p.numel() for p in trainable),
               "injection": args.injection, "mask_mode": args.mask_mode,
               "freeze_config": args.freeze_config}
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({**summary, "history": history}, f, indent=2)
    print(json.dumps(summary, indent=2))


@torch.no_grad()
def run_eval(vlm, val_dl, injection, device, max_examples, max_new_tokens) -> float:
    from vlm.eval import batch_clevr_accuracy
    from vlm.textbatch import make_prompt

    vlm.eval()
    preds, golds, qtypes, seen = [], [], [], 0
    for batch in val_dl:
        prompts = [make_prompt(q, injection) for q in batch["question"]]
        outs = vlm.generate(batch["image"].to(device), prompts,
                            injection=injection, max_new_tokens=max_new_tokens, do_sample=False)
        preds.extend(outs)
        golds.extend(batch["answer"])
        qtypes.extend(batch["q_type"])
        seen += len(batch["answer"])
        if seen >= max_examples:
            break
    return batch_clevr_accuracy(preds, golds, qtypes)["overall"]


def save_checkpoint(args, vlm, vit_config, pos_encoding, decoder_name, image_token_id) -> None:
    torch.save(
        {
            "vit_config": vit_config,
            "pos_encoding": pos_encoding,
            "decoder_model_name": decoder_name,
            "injection": args.injection,
            "mask_mode": args.mask_mode,
            "freeze_config": args.freeze_config,
            "image_token_id": image_token_id,
            "vit": vlm.vit.state_dict(),
            "projector": vlm.projector.state_dict(),
            "decoder": vlm.decoder.state_dict(),
        },
        args.output_dir / "best.pt",
    )


if __name__ == "__main__":
    main()
