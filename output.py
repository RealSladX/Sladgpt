import sys
import time
import os
from collections.abc import Iterator
import torch

def fancy_print(name: str):
    node_name = name
    print_string = ""
    for l in node_name:
        if l == " ":
            print_string += chr(32)
            _ = sys.stdout.write(f"{print_string}\n")
            _ = sys.stdout.write("\x1b[1A")
            _ = sys.stdout.write("\x1b[2K")
        else:
            print_letter=33
            while chr(print_letter) != l:
                print_letter += 1
                time.sleep(0.00000005)
                _ = sys.stdout.write(f"{print_string}{chr(print_letter)}\n")
                _ = sys.stdout.write("\x1b[1A")
                _ = sys.stdout.write("\x1b[2K")
            print_string += chr(print_letter)
            _ =sys.stdout.write(f"{print_string}\n")
            _ =sys.stdout.write("\x1b[1A")
            _ =sys.stdout.write("\x1b[2K")
    _ = sys.stdout.write(f"{print_string}\n")

def print_iterate_files(files: Iterator[os.DirEntry]):
    for file in files:
        file_size = round(file.stat()[6] / 1e9, 3)
        fancy_print(f"{file.name} {str(file_size)} GB")

def check_torch():
    fancy_print(f"PyTorch version: {torch.__version__}")
    fancy_print(f"CUDA available: {torch.cuda.is_available()}")
    fancy_print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        fancy_print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        fancy_print("No GPU")
    fancy_print(f"CUDA version in PyTorch: {torch.version.cuda}")

def print_x_y(f, x, y, block_size):
    for k in range(block_size):
        context = x[:k+1]
        targets = y[k]
        fancy_print(f"When input is {f(context.to('cpu').numpy())} the target is {f([targets.item()])}")
