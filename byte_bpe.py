import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import regex as re

# GPT-2 style pretokenization pattern.
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """
    Reversible byte->unicode map used by GPT-2 byte-level BPE.
    This version covers the full 0..255 byte range.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    prev_char = word[0]
    for ch in word[1:]:
        pairs.add((prev_char, ch))
        prev_char = ch
    return pairs


@dataclass
class ByteBPETokenizer:
    encoder: Dict[str, int]
    merges: List[Tuple[str, str]]
    errors: str = "replace"

    def __post_init__(self) -> None:
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.cache: Dict[str, str] = {}
        self.pat = re.compile(GPT2_PATTERN)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def eot_token(self) -> int:
        # GPT-2 commonly uses <|endoftext|> as the end-of-document token.
        return self.encoder["<|endoftext|>"]

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: List[str] = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str, add_prefix_space: bool = False) -> List[int]:
        if add_prefix_space and text and not text.startswith(" "):
            text = " " + text

        out: List[int] = []
        for token in re.findall(self.pat, text):
            mapped = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            pieces = self.bpe(mapped).split(" ")
            out.extend(self.encoder[p] for p in pieces)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        text = "".join(self.decoder[int(i)] for i in ids)
        byte_arr = bytearray(self.byte_decoder[c] for c in text)
        return byte_arr.decode("utf-8", errors=self.errors)

    @staticmethod
    def load(vocab_json_path: str, merges_path: str, errors: str = "replace") -> "ByteBPETokenizer":
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            encoder = json.load(f)

        merges: List[Tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((a, b))

        return ByteBPETokenizer(encoder=encoder, merges=merges, errors=errors)

    def save(self, vocab_json_path: str, merges_path: str) -> None:
        with open(vocab_json_path, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

