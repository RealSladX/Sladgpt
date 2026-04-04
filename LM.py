import os
from output import fancy_print, check_torch, print_x_y
from utils import pretokenize, train_val_split, estimate_loss, bytes_to_unicode
import torch
from pprint import pp
from gpt import GPTLanguageModel
import pickle
from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, "data")
data_files = os.scandir(DATA_PATH)
MODEL_PATH = os.path.join(ABS_PATH, "models")

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
# output.fancy_print("Identifying available data...")
# output.print_iterate_files(data_files)



# fancy_print(f"Encoding {file_name}")

# enc_tensor, vocab_size, stoi, itos = pretokenize(os.path.join(DATA_PATH, f"{file_name}"))
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])
# tensor_size = str(round(len(enc_tensor) / 1e9, 3))
# fancy_print(f"Successfully created tensor Size: {tensor_size} GB")
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
