import math
import os
from pprint import pp
import torch
from data_utils import build_dataset_paths, BatchProvider, estimate_loss
from gpt import GPTLanguageModel
from byte_bpe import ByteBPETokenizer
from output import check_torch, fancy_print
from parameters import (
    DATA_bin_dir,
    DATA_prefix,
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
    MODEL_test_prompts,
    MODEL_weight_decay,
    TOKENIZER_merges_txt,
    TOKENIZER_vocab_json,
)


block_size = MODEL_block_size
batch_size = MODEL_batch_size
max_iters = MODEL_max_iters
eval_iters = MODEL_eval_iters
eval_interval = MODEL_eval_interval
learning_rate = MODEL_learning_rate
weight_decay = MODEL_weight_decay
grad_clip = MODEL_grad_clip
n_embeddings = MODEL_n_embeddings
n_head = MODEL_n_head
n_decoder_layers = MODEL_n_decoder_layers
dropout = MODEL_dropout

check_torch()
device = "cuda" if torch.cuda.is_available() else "cpu"
fancy_print(f"Device set to: {device}")
paths = build_dataset_paths(DATA_bin_dir, DATA_prefix)

train_mm = paths.train.open()
val_mm = paths.val.open()
vocab_size = paths.vocab_size
batch_provider = BatchProvider(train_mm, val_mm, block_size, batch_size, device)

tokenizer = ByteBPETokenizer.load(TOKENIZER_vocab_json, TOKENIZER_merges_txt)
encode = lambda s: tokenizer.encode(s)
decode = lambda ids: tokenizer.decode(ids)

fancy_print(f"Utilitzing GPT Language Model")
fancy_print(
    f"Block size: {block_size}, Embedding Vector Size: {n_embeddings}, Number of Attention Heads: {n_head}, Number of Decoder Layers: {n_decoder_layers},"
)
bmodel = GPTLanguageModel(
    vocab_size, block_size, n_embeddings, n_head, n_decoder_layers, dropout
)
m = bmodel.to(device)

# Measure Initial Model Performance
optimizer = torch.optim.AdamW(
    m.parameters(), lr=learning_rate, weight_decay=weight_decay
)

base_lr = learning_rate
min_lr = learning_rate * 0.1


def get_lr(iter_num: int) -> float:
    if iter_num >= max_iters:
        return min_lr

    decay_ratio = iter_num / max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def set_lr(iter_num: int) -> float:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


ckpt_name = (
    f"{DATA_prefix}_"
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
        fancy_print(f"Resuming from iteration {start_iter}")
    else:
        fancy_print("Checkpoint architecture mismatch; starting fresh")
else:
    fancy_print("Architecture not found. Starting Fresh.")

last_iter = start_iter - 1

for iter in range(start_iter, max_iters + 1):
    last_iter = iter
    lr_now = set_lr(iter)
    if iter % eval_interval == 0:
        losses = estimate_loss(m, batch_provider, eval_iters)
        fancy_print(
            f"{iter + 1} training loss: {losses['train']:.2f} validation loss: {losses['val']:.2f} lr: {lr_now:.2e}"
        )

    train_x, train_y = batch_provider.get_batch("train")
    logits, loss = m(train_x, train_y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
    optimizer.step()

for prompt in MODEL_test_prompts:
    encoded_input = encode(prompt)
    context = torch.tensor([encoded_input], dtype=torch.long, device=device)
    generated_chars = decode(
        m.generate(
            context,
            300,
            temperature=0.85,
            top_k=120,
            top_p=0.92,
            repetition_penalty=1.1,
            penalty_window=64,
        )[0].tolist()
    )
    pp(f"When input is {decode(context.to('cpu').numpy()[0])} the output is:")
    pp(f"{generated_chars}")

fancy_print("Saving Model...")


torch.save(
    {
        "iter": last_iter,
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_embeddings": n_embeddings,
        "n_head": n_head,
        "n_decoder_layers": n_decoder_layers,
        "dropout": dropout,
    },
    ckpt_path,
)

fancy_print("Model Saved!")
