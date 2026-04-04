import torch

def bytes_to_unicode():
    bs = (
            list(range(ord("!"), ord("~") + 1))
            )
    cs = bs
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for ch in word[1:]:
        pairs.add((prev_char, ch))
        prev_char = ch
    return pairs

def encode(data, stoi):
    return [stoi[c] for c in data]

def pretokenize(file_path:str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    chars = sorted(set(data))
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}

    data = torch.tensor(encode(data, stoi), dtype=torch.long)
    return data, len(chars), stoi, itos

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
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
