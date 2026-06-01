"""Helpers for building tokenized CLEVR QA batches for the VLM (§5).

Not a PROVIDED file — added so train_vlm.py and eval_vlm.py share the same
prompt formatting and answer-only label masking.
"""

from __future__ import annotations

import torch

IMAGE_TOKEN = "<image>"


def make_prompt(question: str, injection: str) -> str:
    """Text prompt fed to the decoder (the answer is appended only at train time).

    For interleaved injection the prompt carries the <image> placeholder; for
    cls / all_patches the visual tokens are prepended by the model, so the
    prompt is just the question.
    """
    prefix = f"{IMAGE_TOKEN}\n" if injection == "interleaved" else ""
    return f"{prefix}Question: {question}\nAnswer:"


def build_qa_batch(tokenizer, questions, answers, injection, device, max_len=64):
    """Tokenize (question, answer) pairs with answer-only loss masking.

    Returns (input_ids, attention_mask, labels) tensors on `device`.
    Prompt + padding positions are set to -100 in labels so only answer tokens
    contribute to the loss.
    """
    eos = tokenizer.eos_token or ""
    seqs, label_seqs = [], []
    for q, a in zip(questions, answers):
        prompt_ids = tokenizer(make_prompt(q, injection), add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(" " + str(a) + eos, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        seqs.append(ids)
        label_seqs.append(labels)

    T = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    B = len(seqs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(seqs, label_seqs)):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(lab)] = torch.tensor(lab)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)
