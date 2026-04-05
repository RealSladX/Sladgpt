from __future__ import annotations
import json
import argparse
import hashlib
import os
from typing import Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from data_utils import choose_memmap_dtype
from byte_bpe import ByteBPETokenizer


def stable_bucket(text: str, modulo: int = 10_000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
    return int(digest, 16) % modulo


def route_by_hash(text: str, val_ratio: float) -> str:
    threshold = int(val_ratio * 10_000)
    return "val" if stable_bucket(text) < threshold else "train"


def write_ids(sink, ids, np_dtype) -> None:
    arr = np.asarray(ids, dtype=np_dtype)
    sink.write(arr.tobytes())


def encode_record(tokenizer: ByteBPETokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_prefix_space=False)
    ids.append(tokenizer.eot_token)
    return ids


def prepare_from_hf(
    dataset_name: str,
    text_key: str,
    out_dir: str,
    out_prefix: str,
    tokenizer: ByteBPETokenizer,
    val_ratio: float,
    max_examples: Optional[int],
) -> None:
    dtype_name = choose_memmap_dtype(tokenizer.vocab_size)
    np_dtype = np.uint16 if dtype_name == "uint16" else np.uint32

    train_path = os.path.join(out_dir, f"{out_prefix}_train.bin")
    val_path = os.path.join(out_dir, f"{out_prefix}_val.bin")
    meta_path = os.path.join(out_dir, f"{out_prefix}_meta.json")
    os.makedirs(out_dir, exist_ok=True)

    count_train = 0
    count_val = 0
    tokens_train = 0
    tokens_val = 0
    processed = 0

    with open(train_path, "wb") as train_f, open(val_path, "wb") as val_f:
        if dataset_name == "openwebtext":
            ds = load_dataset("Skylion007/openwebtext", "plain_text", split="train", streaming=True)
            iterable = ds
            for ex in tqdm(iterable, desc=f"encoding {dataset_name}"):
                if max_examples is not None and processed >= max_examples:
                    break
                text = ex[text_key]
                if not text:
                    continue
                ids = encode_record(tokenizer, text)
                split = route_by_hash(text, val_ratio)
                if split == "train":
                    write_ids(train_f, ids, np_dtype)
                    count_train += 1
                    tokens_train += len(ids)
                else:
                    write_ids(val_f, ids, np_dtype)
                    count_val += 1
                    tokens_val += len(ids)
                processed += 1

        elif dataset_name == "tinystories":
            # Prefer explicit train/validation splits if available.
            try:
                train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
                val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

                for ex in tqdm(train_ds, desc="encoding tinystories train"):
                    if max_examples is not None and processed >= max_examples:
                        break
                    text = ex[text_key]
                    if not text:
                        continue
                    ids = encode_record(tokenizer, text)
                    write_ids(train_f, ids, np_dtype)
                    count_train += 1
                    tokens_train += len(ids)
                    processed += 1

                for ex in tqdm(val_ds, desc="encoding tinystories val"):
                    if max_examples is not None and processed >= max_examples:
                        break
                    text = ex[text_key]
                    if not text:
                        continue
                    ids = encode_record(tokenizer, text)
                    write_ids(val_f, ids, np_dtype)
                    count_val += 1
                    tokens_val += len(ids)
                    processed += 1

            except Exception:
                ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
                for ex in tqdm(ds, desc="encoding tinystories"):
                    if max_examples is not None and processed >= max_examples:
                        break
                    text = ex[text_key]
                    if not text:
                        continue
                    ids = encode_record(tokenizer, text)
                    split = route_by_hash(text, val_ratio)
                    if split == "train":
                        write_ids(train_f, ids, np_dtype)
                        count_train += 1
                        tokens_train += len(ids)
                    else:
                        write_ids(val_f, ids, np_dtype)
                        count_val += 1
                        tokens_val += len(ids)
                    processed += 1
        else:
            raise ValueError(f"Unsupported dataset_name={dataset_name}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "vocab_size": tokenizer.vocab_size,
                "eot_token": tokenizer.eot_token,
                "dtype": dtype_name,
                "train_examples": count_train,
                "val_examples": count_val,
                "train_tokens": tokens_train,
                "val_tokens": tokens_val,
                "train_bin": train_path,
                "val_bin": val_path,
            },
            f,
            indent=2,
        )
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")
    print(f"wrote {meta_path}")
    print(f"train examples={count_train} tokens={tokens_train}")
    print(f"val examples={count_val} tokens={tokens_val}")


def prepare_from_text_files(
    train_text_path: str,
    val_text_path: Optional[str],
    out_dir: str,
    out_prefix: str,
    tokenizer: ByteBPETokenizer,
    val_ratio: float,
) -> None:
    dtype_name = choose_memmap_dtype(tokenizer.vocab_size)
    np_dtype = np.uint16 if dtype_name == "uint16" else np.uint32

    train_path = os.path.join(out_dir, f"{out_prefix}_train.bin")
    val_path = os.path.join(out_dir, f"{out_prefix}_val.bin")
    meta_path = os.path.join(out_dir, f"{out_prefix}_meta.json")
    os.makedirs(out_dir, exist_ok=True)

    with open(train_text_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    if val_text_path is not None:
        with open(val_text_path, "r", encoding="utf-8") as f:
            val_text = f.read()
    else:
        n = int((1.0 - val_ratio) * len(train_text))
        train_text, val_text = train_text[:n], train_text[n:]

    train_ids = encode_record(tokenizer, train_text)
    val_ids = encode_record(tokenizer, val_text)

    with open(train_path, "wb") as f:
        write_ids(f, train_ids, np_dtype)
    with open(val_path, "wb") as f:
        write_ids(f, val_ids, np_dtype)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "vocab_size": tokenizer.vocab_size,
                "eot_token": tokenizer.eot_token,
                "dtype": dtype_name,
                "train_examples": count_train,
                "val_examples": count_val,
                "train_tokens": tokens_train,
                "val_tokens": tokens_val,
                "train_bin": train_path,
                "val_bin": val_path,
            },
            f,
            indent=2,
        )
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")
    print(f"wrote {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hf", "text"], required=True)
    parser.add_argument("--dataset", choices=["openwebtext", "tinystories"], default=None)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--train_text_path", default=None)
    parser.add_argument("--val_text_path", default=None)
    parser.add_argument("--out_dir", default="data_bin")
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--vocab_json", required=True)
    parser.add_argument("--merges_txt", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.001)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer.load(args.vocab_json, args.merges_txt)

    if args.mode == "hf":
        if args.dataset is None:
            raise ValueError("--dataset is required when mode=hf")
        prepare_from_hf(
            dataset_name=args.dataset,
            text_key=args.text_key,
            out_dir=args.out_dir,
            out_prefix=args.out_prefix,
            tokenizer=tokenizer,
            val_ratio=args.val_ratio,
            max_examples=args.max_examples,
        )
    else:
        if args.train_text_path is None:
            raise ValueError("--train_text_path is required when mode=text")
        prepare_from_text_files(
            train_text_path=args.train_text_path,
            val_text_path=args.val_text_path,
            out_dir=args.out_dir,
            out_prefix=args.out_prefix,
            tokenizer=tokenizer,
            val_ratio=args.val_ratio,
        )


if __name__ == "__main__":
    main()

