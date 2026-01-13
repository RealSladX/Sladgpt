import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pp

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size).to('cuda')

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

        #If there are no targets, then there is no loss
        if targets is None:
            loss = None

        else:
            #Batch, Time, Channels
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is the array of indices in the current Batch at the current Time
        pp(self.token_embedding_table)
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus on the last time step
            logits = logits[:, -1, :] # (Batch, Channels)
            probs = F.softmax(logits, dim=-1)
            # from the probabilities, select the most likely as the next sequence
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)

        return index
