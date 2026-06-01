"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions must be set to -100 in labels
                        before being passed in (so they're masked out by HF's
                        loss).
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_visual(self, images: torch.Tensor, injection: InjectionMode) -> torch.Tensor:
        """Run the ViT + projector. Returns visual tokens (B, N_vis, d_decoder)."""
        if injection == "cls":
            feats = self.vit(images)                       # (B, d_image)
        else:  # all_patches / interleaved use the full token sequence
            feats = self.vit(images, return_all_tokens=True)  # (B, N+1, d_image)
        visual = self.projector(feats)                     # (B, N_vis, d_decoder)
        embed_dtype = self.decoder.get_input_embeddings().weight.dtype
        return visual.to(embed_dtype)

    def _prefix_inject(self, visual, text_embeds, attention_mask, labels):
        """Prepend visual tokens to the text sequence (cls / all_patches)."""
        B, n_vis, _ = visual.shape
        inputs_embeds = torch.cat([visual, text_embeds], dim=1)
        vis_mask = torch.ones(B, n_vis, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            vis_labels = torch.full((B, n_vis), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([vis_labels, labels], dim=1)
        return inputs_embeds, full_mask, full_labels, n_vis

    def _interleave_inject(self, visual, input_ids, text_embeds, attention_mask, labels):
        """Replace the single <image> placeholder token with the visual sequence."""
        assert self.image_token_id is not None, "interleaved mode needs image_token_id"
        B = input_ids.shape[0]
        embed_dtype = text_embeds.dtype
        rows_e, rows_m, rows_l = [], [], []
        for b in range(B):
            pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            assert len(pos) == 1, "expected exactly one <image> token per example"
            p = int(pos[0])
            te, am = text_embeds[b], attention_mask[b]
            e = torch.cat([te[:p], visual[b], te[p + 1:]], dim=0)
            n_vis = visual.shape[1]
            m = torch.cat([
                am[:p],
                torch.ones(n_vis, dtype=am.dtype, device=am.device),
                am[p + 1:],
            ], dim=0)
            rows_e.append(e)
            rows_m.append(m)
            if labels is not None:
                lb = labels[b]
                l = torch.cat([
                    lb[:p],
                    torch.full((n_vis,), -100, dtype=lb.dtype, device=lb.device),
                    lb[p + 1:],
                ], dim=0)
                rows_l.append(l)
        # Right-pad to the longest stitched sequence.
        T = max(e.shape[0] for e in rows_e)
        def pad(seq, value, dtype):
            out = torch.full((B, T, *seq[0].shape[1:]), value, dtype=dtype, device=seq[0].device)
            for b, s in enumerate(seq):
                out[b, : s.shape[0]] = s
            return out
        inputs_embeds = pad(rows_e, 0.0, embed_dtype)
        full_mask = pad(rows_m, 0, attention_mask.dtype)
        full_labels = pad(rows_l, -100, labels.dtype) if labels is not None else None
        return inputs_embeds, full_mask, full_labels

    def _build_inputs(self, images, input_ids, attention_mask, labels, injection):
        visual = self._encode_visual(images, injection)
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        if injection == "interleaved":
            inputs_embeds, full_mask, full_labels = self._interleave_inject(
                visual, input_ids, text_embeds, attention_mask, labels
            )
            n_vis = visual.shape[1]  # contiguous block per example (used only for prefix bidir)
        else:
            inputs_embeds, full_mask, full_labels, n_vis = self._prefix_inject(
                visual, text_embeds, attention_mask, labels
            )
        return inputs_embeds, full_mask, full_labels, n_vis

    def _attention_mask(self, full_mask, n_vis, mask_mode, dtype):
        """Return the attention mask to pass to the decoder.

        - causal: the 2D padding mask (decoder adds its own causal mask).
        - image_bidir: a 4D additive mask, bidirectional inside the visual prefix
          and causal everywhere else, merged with padding.
        """
        if mask_mode == "causal":
            return full_mask
        B, T = full_mask.shape
        device = full_mask.device
        base = build_image_bidir_mask(n_vis, T - n_vis, device, dtype)  # (1,1,T,T)
        # Merge padding: padded key positions are never attended to.
        pad_add = torch.where(
            full_mask[:, None, None, :].bool(),
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
        )  # (B,1,1,T)
        return base + pad_add  # (B,1,T,T)

    # ------------------------------------------------------------------
    # Forward / generate
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        inputs_embeds, full_mask, full_labels, n_vis = self._build_inputs(
            images, input_ids, attention_mask, labels, injection
        )
        attn = self._attention_mask(full_mask, n_vis, mask_mode, inputs_embeds.dtype)
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=full_labels,
        )
        out = {"logits": outputs.logits}
        if full_labels is not None:
            out["loss"] = outputs.loss
        return out

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Useful for §5's qualitative evaluation problem (vlm_qualitative).
        """
        device = next(self.decoder.parameters()).device
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        inputs_embeds, full_mask, _, _ = self._build_inputs(
            images.to(device), input_ids, attention_mask, labels=None, injection=injection
        )
        gen = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
