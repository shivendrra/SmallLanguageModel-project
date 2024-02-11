import torch
import numpy as np

class nnMods:
 
    @staticmethod
    def token_embeddings(vocab_size, n_embd):
        token_embeddings = torch.randn(vocab_size, n_embd)
        token_embeddings /= torch.norm(token_embeddings, dim=1, keepdim=True)
        return token_embeddings
    
    @staticmethod
    def position_embeddings(block_size, n_embd):
        position_embeddings = torch.zeros(block_size, n_embd)
        for pos in range(block_size):
            for i in range(0, n_embd, 2):
                position = torch.tensor(pos, dtype=torch.float)
                exponent = torch.tensor(i / n_embd, dtype=torch.float)
                position_embeddings[pos, i] = torch.sin(position / (10000 ** exponent))
                position_embeddings[pos, i + 1] = torch.cos(position / (10000 ** exponent))
        return position_embeddings
    
    @staticmethod
    def sequential(*module):
        for module in module:
            x = module(x)
        return x



class CustomTransformerModel(nnMods):
    def __init__(self, vocab_size, block_size, n_embd):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding_table = nnMods.token_embeddings(self.vocab_size, self.n_embd)
        self.position_embedding_table = nnMods.position_embeddings(self.block_size, self.n_embd)


    def forward(self, idx):
        tok_emb = self.token_embedding_table[idx]  # (B, T, C)
        pos_emb = self.position_embedding_table[:tok_emb.shape[1]]  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        return x

batch_size = 64
vocab_size = 483
block_size = 128
n_embd = 8

model = CustomTransformerModel(vocab_size, block_size, n_embd)
idx = torch.randint(0, vocab_size, (batch_size, block_size))
token_embeddings = model.token_embedding_table
position_embeddings = model.position_embedding_table
output = model.forward(idx)

print("inputs: ", idx.shape)
print("Token Embeddings Shape:", token_embeddings.shape)
print("Positional Embeddings Shape:", position_embeddings.shape)
print("Output Shape:", output.shape)