from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from byte_bpe import ByteBPETokenizer
from data_utils import choose_memmap_dtype


def write_ids(sink, ids: list[int], np_dtype) -> int:
    arr = np.asarray(ids, dtype=np_dtype)
    sink.write(arr.tobytes())
    return len(ids)


def encode_text_file(
    tokenizer: ByteBPETokenizer,
    in_path: str,
    out_path: str,
    np_dtype,
    chunk_size: int,
) -> int:
    total_tokens = 0

    with (
        open(in_path, "r", encoding="utf-8", errors="replace") as src,
        open(out_path, "wb") as dst,
    ):
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break

            ids = tokenizer.encode(chunk)
            total_tokens += write_ids(dst, ids, np_dtype)

        total_tokens += write_ids(dst, [tokenizer.eot_token], np_dtype)

    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_text_path", required=True)
    parser.add_argument("--val_text_path", required=True)
    parser.add_argument("--out_dir", default="data_bin")
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--vocab_json", required=True)
    parser.add_argument("--merges_txt", required=True)
    parser.add_argument("--chunk_size", type=int, default=1048576)
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer.load(args.vocab_json, args.merges_txt)

    dtype_name = choose_memmap_dtype(tokenizer.vocab_size)
    np_dtype = np.uint16 if dtype_name == "uint16" else np.uint32

    os.makedirs(args.out_dir, exist_ok=True)

    train_bin = os.path.join(args.out_dir, f"{args.out_prefix}_train.bin")
    val_bin = os.path.join(args.out_dir, f"{args.out_prefix}_val.bin")
    meta_path = os.path.join(args.out_dir, f"{args.out_prefix}_meta.json")

    train_tokens = encode_text_file(
        tokenizer, args.train_text_path, train_bin, np_dtype, args.chunk_size
    )
    val_tokens = encode_text_file(
        tokenizer, args.val_text_path, val_bin, np_dtype, args.chunk_size
    )

    meta = {
        "dataset_name": "text_bpe",
        "vocab_size": tokenizer.vocab_size,
        "eot_token": tokenizer.eot_token,
        "dtype": dtype_name,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_bin": train_bin,
        "val_bin": val_bin,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {train_bin}")
    print(f"wrote {val_bin}")
    print(f"wrote {meta_path}")
    print(f"train_tokens={train_tokens}")
    print(f"val_tokens={val_tokens}")
    print(f"vocab_size={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
