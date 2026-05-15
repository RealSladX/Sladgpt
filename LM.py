import os
from pprint import pp
import torch
from data_utils import build_dataset_paths, BatchProvider, estimate_loss
from gpt import GPTLanguageModel
from output import check_torch, fancy_print
from parameters import (
    MODEL_batch_size,
    MODEL_block_size,
    MODEL_dropout,
    MODEL_eval_interval,
    MODEL_eval_iters,
    MODEL_grad_clip,
    MODEL_learning_rate,
    MODEL_max_iters,
    MODEL_n_decoder_layers,
    MODEL_n_embeddings,
    MODEL_n_head,
    MODEL_test_prompt,
    MODEL_weight_decay,
)



PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


block_size = MODEL_block_size
batch_size = MODEL_batch_size
max_iters = MODEL_max_iters
eval_iters = MODEL_eval_iters
eval_interval = MODEL_eval_interval
learning_rate= MODEL_learning_rate
weight_decay = MODEL_weight_decay
grad_clip = MODEL_grad_clip
n_embeddings = MODEL_n_embeddings
n_head = MODEL_n_head
n_decoder_layers = MODEL_n_decoder_layers
dropout = MODEL_dropout

check_torch()
device = "cuda" if torch.cuda.is_available() else "cpu"

paths = build_dataset_paths("data_bin", "tinystories_char")
train_mm = paths.train.open()
val_mm = paths.val.open()
vocab_size = paths.vocab_size
batch_provider = BatchProvider(train_mm, val_mm, block_size, batch_size, device)


stoi = paths.meta["stoi"]
itos = {int(k): v for k, v in paths.meta["itos"].items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

fancy_print(f"Utilitzing GPT Language Model")
fancy_print(f"Block size: {block_size}, Embedding Vector Size: {n_embeddings}, Number of Attention Heads: {n_head}, Number of Decoder Layers: {n_decoder_layers},")
bmodel = GPTLanguageModel(vocab_size, block_size, n_embeddings, n_head, n_decoder_layers, dropout)
m = bmodel.to(device)

# Measure Initial Model Performance
encoded_input = encode(MODEL_test_prompt)
context = torch.tensor([encoded_input], dtype=torch.long, device=device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_iters,
    eta_min=learning_rate * 0.1,
)
ckpt_name = (
    f"tinystories_char_"
    f"bs{block_size}_"
    f"bt{batch_size}_"
    f"emb{n_embeddings}_"
    f"h{n_head}_"
    f"l{n_decoder_layers}.pt"
)

ckpt_path = os.path.join("./models", ckpt_name)

start_iter = 0

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)

    if (
        ckpt["vocab_size"] == vocab_size
        and ckpt["block_size"] == block_size
        and ckpt["n_embeddings"] == n_embeddings
        and ckpt["n_head"] == n_head
        and ckpt["n_decoder_layers"] == n_decoder_layers
        and ckpt["dropout"] == dropout
    ):
        m.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iter"] + 1
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        fancy_print(f"Resuming from iteration {start_iter}")
    else:
        fancy_print("Checkpoint architecture mismatch; starting fresh")

generated_chars = decode(m.generate(context, 500, temperature=0.7, top_k=40)[0].tolist())
fancy_print(f"Current model performance")
pp(f"When input is {decode(context.to('cpu').numpy()[0])} the output is:")
pp(f"{generated_chars}")


last_iter = start_iter - 1

for iter in range(start_iter, max_iters + 1):
    last_iter = iter
    if iter % eval_interval == 0:
        losses = estimate_loss(m, batch_provider, eval_iters)
        fancy_print(f"{iter+1} training loss: {losses['train']:.2f} validation loss: {losses['val']:.2f}")

    train_x, train_y = batch_provider.get_batch("train")
    logits, loss = m(train_x, train_y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()

generated_chars = decode(m.generate(context, 500,temperature=0.7, top_k=40)[0].tolist())
fancy_print(f"Model performance after {max_iters} iterations")
pp(f"When input is {decode(context.to('cpu').numpy()[0])} the output is:")
pp(f"{generated_chars}")

fancy_print("Saving Model...")


torch.save({
    "iter": last_iter,
    "model_state_dict": m.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embeddings": n_embeddings,
    "n_head": n_head,
    "n_decoder_layers": n_decoder_layers,
    "dropout": dropout,
}, ckpt_path)

fancy_print('Model Saved!')
