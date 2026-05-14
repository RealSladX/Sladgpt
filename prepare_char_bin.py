import json
import os
import numpy as np

DATA_PATH = "data"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"
OUT_DIR = "data_bin"
PREFIX = "tinystories_char"

os.makedirs(OUT_DIR, exist_ok=True)

# pass 1: build vocab from train + val without storing full files
chars = set()
for name in [TRAIN_FILE, VAL_FILE]:
    with open(os.path.join(DATA_PATH, name), "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            chars.update(chunk)

chars = sorted(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

dtype = np.uint16 if len(chars) <= np.iinfo(np.uint16).max else np.uint32

def encode_file(in_name, out_name):
    out_path = os.path.join(OUT_DIR, out_name)
    with open(os.path.join(DATA_PATH, in_name), "r", encoding="utf-8") as src, open(out_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            arr = np.fromiter((stoi[c] for c in chunk), dtype=dtype, count=len(chunk))
            dst.write(arr.tobytes())
    return out_path

train_bin = encode_file(TRAIN_FILE, f"{PREFIX}_train.bin")
val_bin = encode_file(VAL_FILE, f"{PREFIX}_val.bin")

meta = {
    "dataset_name": "tinystories_char",
    "vocab_size": len(chars),
    "dtype": "uint16" if dtype == np.uint16 else "uint32",
    "stoi": stoi,
    "itos": {str(k): v for k, v in itos.items()},
    "train_bin": train_bin,
    "val_bin": val_bin,
}

with open(os.path.join(OUT_DIR, f"{PREFIX}_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)

print("wrote", train_bin)
print("wrote", val_bin)
print("vocab_size", len(chars))
