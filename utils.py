import torch

def pretokenize(file_path:str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    chars = sorted(set(data))
    vocab_size = len(chars)
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(data), dtype=torch.long).to("cuda")
    return data, vocab_size, stoi, itos

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to("cuda")
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to("cuda")
    return x, y

def train_val_split(data, block_size, batch_size):
    n = int(0.8*len(data))
    train_data = data[:n]
    train_x, train_y = get_batch(train_data, block_size, batch_size)
    val_data = data[n:]
    val_x, val_y = get_batch(val_data, block_size, batch_size)
    return train_x, train_y, val_x, val_y

@torch.no_grad()
def estimate_loss(model, eval_iters, data, block_size, batch_size):
    out = {}
    model.eval()
    losses_train = torch.zeros(eval_iters).to("cuda")
    losses_val = torch.zeros(eval_iters).to("cuda")
    for k in range(eval_iters):
        train_x, train_y, val_x, val_y = train_val_split(data, block_size, batch_size)
        logits, loss = model(train_x, train_y)
        losses_train[k] = loss.item()
        logits, loss = model(val_x, val_y)
        losses_val[k] = loss.item()
    out['train'] = losses_train.mean()
    out['val'] = losses_val.mean()
    model.train()
    return out
