from __future__ import annotations

import argparse
import collections
import os
from typing import Iterable

import regex as re
from tqdm import tqdm

from byte_bpe import ByteBPETokenizer, GPT2_PATTERN, bytes_to_unicode, get_pairs


def iter_chunks(paths: list[str], chunk_size: int) -> Iterable[str]:
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk


def build_word_counts(paths: list[str], chunk_size: int) -> collections.Counter[tuple[str, ...]]:
    pat = re.compile(GPT2_PATTERN)
    byte_encoder = bytes_to_unicode()
    word_counts: collections.Counter[tuple[str, ...]] = collections.Counter()

    for chunk in tqdm(iter_chunks(paths, chunk_size), desc="pretokenizing"):
        for token in re.findall(pat, chunk):
            mapped = "".join(byte_encoder[b] for b in token.encode("utf-8"))
            word_counts[tuple(mapped)] += 1

    return word_counts


def most_frequent_pair(word_counts: collections.Counter[tuple[str, ...]]) -> tuple[str, str] | None:
    pair_counts: collections.Counter[tuple[str, str]] = collections.Counter()

    for word, count in word_counts.items():
        for pair in get_pairs(word):
            pair_counts[pair] += count

    if not pair_counts:
        return None

    return pair_counts.most_common(1)[0][0]


def merge_pair_in_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    first, second = pair
    out: list[str] = []
    i = 0

    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            out.append(first + second)
            i += 2
        else:
            out.append(word[i])
            i += 1

    return tuple(out)


def merge_pair(
    word_counts: collections.Counter[tuple[str, ...]],
    pair: tuple[str, str],
) -> collections.Counter[tuple[str, ...]]:
    new_counts: collections.Counter[tuple[str, ...]] = collections.Counter()

    for word, count in word_counts.items():
        new_word = merge_pair_in_word(word, pair)
        new_counts[new_word] += count

    return new_counts


def train_bpe(paths: list[str], vocab_size: int, chunk_size: int) -> ByteBPETokenizer:
    byte_encoder = bytes_to_unicode()

    encoder: dict[str, int] = {v: i for i, v in enumerate(byte_encoder.values())}
    encoder["<|endoftext|>"] = len(encoder)

    word_counts = build_word_counts(paths, chunk_size)
    merges: list[tuple[str, str]] = []

    target_merges = vocab_size - len(encoder)
    if target_merges <= 0:
        return ByteBPETokenizer(encoder=encoder, merges=merges)

    for _ in tqdm(range(target_merges), desc="learning merges"):
        pair = most_frequent_pair(word_counts)
        if pair is None:
            break

        merged_token = pair[0] + pair[1]
        word_counts = merge_pair(word_counts, pair)

        if merged_token in encoder:
            continue

        merges.append(pair)
        encoder[merged_token] = len(encoder)

    return ByteBPETokenizer(encoder=encoder, merges=merges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_text_path", required=True)
    parser.add_argument("--val_text_path", default=None)
    parser.add_argument("--out_dir", default="tokenizer_out")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=1048576)
    args = parser.parse_args()

    paths = [args.train_text_path]
    if args.val_text_path is not None:
        paths.append(args.val_text_path)

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = train_bpe(paths, args.vocab_size, args.chunk_size)

    vocab_path = os.path.join(args.out_dir, "vocab.json")
    merges_path = os.path.join(args.out_dir, "merges.txt")
    tokenizer.save(vocab_path, merges_path)

    print(f"wrote {vocab_path}")
    print(f"wrote {merges_path}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"merges={len(tokenizer.merges)}")

    probe = "Once upon a time, there was a little dog."
    ids = tokenizer.encode(probe)
    decoded = tokenizer.decode(ids)
    print(f"probe ids={ids[:30]}")
    print(f"probe decoded={decoded!r}")

    if decoded != probe:
        raise RuntimeError("Tokenizer round-trip failed")


if __name__ == "__main__":
    main()
