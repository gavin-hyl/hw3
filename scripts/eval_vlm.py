"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy. Useful for both Problem (vlm_qualitative) and Problem (mrope_impl).

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _wrap_decoder_lora(decoder, rank=8, alpha=16.0):
    from basics.lora import LoRALinear

    for p in decoder.parameters():
        p.requires_grad_(False)
    for module in decoder.modules():
        for attr in ("q_proj", "v_proj"):
            child = getattr(module, attr, None)
            if isinstance(child, nn.Linear):
                setattr(module, attr, LoRALinear(child, rank, alpha))
    return decoder


def reconstruct_vlm(ckpt, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from basics.vit import ViT
    from vlm.model import VisionLanguageModel
    from vlm.projector import VisionLanguageProjector
    from vlm.textbatch import IMAGE_TOKEN

    vit = ViT(**ckpt["vit_config"], pos_encoding=ckpt.get("pos_encoding") or "learned")
    decoder = AutoModelForCausalLM.from_pretrained(
        ckpt["decoder_model_name"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt["decoder_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if ckpt["injection"] == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        decoder.resize_token_embeddings(len(tokenizer))
    if ckpt["freeze_config"] == "B":
        _wrap_decoder_lora(decoder)

    projector = VisionLanguageProjector(vit.d_model, decoder.config.hidden_size)
    vlm = VisionLanguageModel(vit, projector, decoder, tokenizer, ckpt["image_token_id"])
    vlm.vit.load_state_dict(ckpt["vit"])
    vlm.projector.load_state_dict(ckpt["projector"])
    vlm.decoder.load_state_dict(ckpt["decoder"])
    vlm.decoder.to(dtype=torch.bfloat16)  # unify (LoRA params reload as fp32 otherwise)
    return vlm.to(device).eval()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    from torchvision.utils import save_image

    from vlm.data import build_clevr_loaders
    from vlm.eval import batch_clevr_accuracy, clevr_exact_match
    from vlm.textbatch import make_prompt

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vlm = reconstruct_vlm(ckpt, device)
    injection = ckpt["injection"]
    vlm.projector.float()

    _, val_dl = build_clevr_loaders(batch_size=32)

    preds, golds, qtypes, dumped, seen = [], [], [], [], 0
    examples_path = args.output_dir / "examples.jsonl"
    with open(examples_path, "w") as fout:
        for batch in val_dl:
            prompts = [make_prompt(q, injection) for q in batch["question"]]
            outs = vlm.generate(batch["image"].to(device), prompts,
                                injection=injection, max_new_tokens=32, do_sample=False)
            for i, pred in enumerate(outs):
                gold = batch["answer"][i]
                correct = clevr_exact_match(pred, gold)
                preds.append(pred); golds.append(gold); qtypes.append(batch["q_type"][i])
                if len(dumped) < args.num_examples:
                    row = {"question": batch["question"][i], "gold": gold,
                           "prediction": pred.strip(), "correct": correct,
                           "q_type": batch["q_type"][i]}
                    if args.save_images:
                        img_path = args.output_dir / f"ex_{len(dumped):02d}.png"
                        save_image(batch["image"][i] * 0.5 + 0.5, img_path)
                        row["image"] = img_path.name
                    fout.write(json.dumps(row) + "\n")
                    dumped.append(row)
            seen += len(outs)
            if seen >= args.max_eval:
                break

    metrics = batch_clevr_accuracy(preds, golds, qtypes)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Accuracy breakdown:", json.dumps(metrics, indent=2))
    print(f"Dumped {len(dumped)} qualitative examples to {examples_path}")


if __name__ == "__main__":
    main()
