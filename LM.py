import json
import math
import os
from pprint import pp
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from byte_bpe import ByteBPETokenizer
from gpt import GPTLanguageModel
from output import check_torch, fancy_print, print_iterate_files
from parameters import (
    DATA_bin_dir,
    DATA_prefix,
    MODEL_batch_size,
    MODEL_block_size,
    MODEL_ckpt_name,
    MODEL_dropout,
    MODEL_eval_interval,
    MODEL_eval_iters,
    MODEL_grad_clip,
    MODEL_learning_rate,
    MODEL_max_iters,
    MODEL_n_decoder_layers,
    MODEL_n_embeddings,
    MODEL_n_head,
    MODEL_sample_tokens_after,
    MODEL_sample_tokens_before,
    MODEL_test_prompt,
    MODEL_weight_decay,
    TOKENIZER_merges_txt,
    TOKENIZER_vocab_json,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, "data")
data_files = os.scandir(DATA_PATH)
MODEL_PATH = os.path.join(ABS_PATH, "models")
vocab_json_path = os.path.join(ABS_PATH, TOKENIZER_vocab_json)
vocab_merges_path = os.path.join(ABS_PATH, TOKENIZER_merges_txt)

block_size = 32
batch_size = 16
max_iters = 10000
eval_iters = 1000
learning_rate=3e-4
weight_decay = 0.1
grad_clip = 1.0
n_embeddings = 512
n_head = 8
n_decoder_layers = 8
dropout = 0.1

file_name = "TinyStoriesV2-GPT4-valid.txt"
test_input = "Thou "

check_torch()


fancy_print("Identifying available data...")
print_iterate_files(data_files)

fancy_print(f"Encoding {file_name}")

data, vocab_size, stoi, itos = pretokenize(os.path.join(DATA_PATH, f"{file_name}"))
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
tensor_size = str(round(len(data) / 1e9, 3))
fancy_print(f"Successfully created tensor Size: {tensor_size} GB")

#
# fancy_print(f"Utilitzing GPT Language Model")
# fancy_print(f"Embedding Vector Size: {n_embeddings}, Number of Attention Heads: {n_head}, Number of Decoder Layers: {n_decoder_layers},")
# bmodel = GPTLanguageModel(vocab_size, block_size, n_embeddings, n_head, n_decoder_layers, dropout)
# m = bmodel.to(device)
#
# # Measure Initial Model Performance
# encoded_input = encode(test_input)
# context = torch.tensor([encoded_input], dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, 500, block_size)[0].tolist())
# fancy_print(f"Initial model performance")
# pp(f"When input is {decode(context.to('cpu').numpy()[0])} the output is:")
# pp(f"{generated_chars}")
#
#
#
# optimizer = torch.optim.AdamW(bmodel.parameters(), lr=learning_rate)
#
# for iter in range(max_iters+1):
#     loss_diff = 0
#     if iter % eval_iters == 0:
#         losses = estimate_loss(m, eval_iters, enc_tensor, block_size, batch_size)
#         fancy_print(f"{iter+1} training loss: {losses['train']:.2f} validation loss: {losses['val']:.2f}")
#
#     train_x, train_y, val_x, val_y = train_val_split(enc_tensor, block_size, batch_size)
#     logits, loss = m.forward(train_x, train_y)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
#
# generated_chars = decode(m.generate(context, 500, block_size)[0].tolist())
# fancy_print(f"Model performance after 5000 iterations")
# pp(f"When input is {decode(context.to('cpu').numpy()[0])} the output is:")
# pp(f"{generated_chars}")
#
# fancy_print("Saving Model...")
#
# with open(os.path.join(MODEL_PATH, 'model-05.pkl'), 'wb') as f:
#     pickle.dump(m, f)
# fancy_print('Model Saved!')
