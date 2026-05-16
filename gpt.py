import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pp
from modules import Block


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.layers = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        b, t = index.shape

        if t > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}; block_size={self.block_size}"
            )
        device = index.device
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(t, device=device))
        x = tok_emb + pos_emb
        x = self.layers(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # If there are no targets, then there is no loss
        if targets is None:
            loss = None

        else:
            # Batch, Time, Channels
            b, t, C = logits.shape
            logits = logits.view(b * t, C)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        index,
        max_new_tokens,
        temperature=0.8,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        penalty_window=64,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            index_cond = index[:, -self.block_size :]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                recent = index[:, -penalty_window:]
                for b in range(logits.size(0)):
                    recent_tokens = recent[b].unique()
                    logits[b, recent_tokens] = (
                        logits[b, recent_tokens] / repetition_penalty
                    )

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)

        return index
