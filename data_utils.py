from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


def choose_memmap_dtype(vocab_size: int) -> str:
    return "uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32"


@dataclass
class BinShard:
    path: str
    dtype: str

    def open(self):
        np_dtype = np.uint16 if self.dtype == "uint16" else np.uint32
        return np.memmap(self.path, dtype=np_dtype, mode="r")


@dataclass
class DatasetPaths:
    train: BinShard
    val: BinShard
    meta: Dict
    vocab_size: int


def build_dataset_paths(data_dir: str, dataset_name: str) -> DatasetPaths:
    meta_json = os.path.join(data_dir, f"{dataset_name}_meta.json")
    meta_pt = os.path.join(data_dir, f"{dataset_name}_meta.pt")

    if os.path.exists(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
    elif os.path.exists(meta_pt):
        meta = torch.load(meta_pt, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find metadata file:\n- {meta_json}\n- {meta_pt}"
        )

    dtype = meta["dtype"]
    train_path = meta.get("train_bin", os.path.join(data_dir, f"{dataset_name}_train.bin"))
    val_path = meta.get("val_bin", os.path.join(data_dir, f"{dataset_name}_val.bin"))

    return DatasetPaths(
        train=BinShard(train_path, dtype),
        val=BinShard(val_path, dtype),
        meta=meta,
        vocab_size=meta["vocab_size"],
    )


class BatchProvider:
    def __init__(self, train_mm, val_mm, block_size: int, batch_size: int, device: str):
        self.train_mm = train_mm
        self.val_mm = val_mm
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str):
        data = self.train_mm if split == "train" else self.val_mm
        if len(data) <= self.block_size + 1:
            raise ValueError(
                f"{split} shard too short: len={len(data)} block_size={self.block_size}"
            )

        ix = torch.randint(0, len(data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(np.asarray(data[i:i+self.block_size], dtype=np.int64))
            for i in ix.tolist()
        ])
        y = torch.stack([
            torch.from_numpy(np.asarray(data[i+1:i+1+self.block_size], dtype=np.int64))
            for i in ix.tolist()
        ])
        return x.to(self.device), y.to(self.device)


@torch.no_grad()
def estimate_loss(model, batch_provider: BatchProvider, eval_iters: int):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=batch_provider.device)
        for k in range(eval_iters):
            xb, yb = batch_provider.get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.detach()
        out[split] = float(losses.mean().item())
    model.train()
    return out
