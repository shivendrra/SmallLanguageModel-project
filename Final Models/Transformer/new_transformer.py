import torch
import numpy as np

class CustomTransformerModel:
    def __init__(self, vocab_size, block_size, n_embd):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding_table = self.initialize_token_embeddings()
        self.position_embedding_table = self.initialize_position_embeddings()

    def initialize_token_embeddings(self):
        token_embeddings = torch.randn(self.vocab_size, self.n_embd)
        token_embeddings /= torch.norm(token_embeddings, dim=1, keepdim=True)
        return token_embeddings

    def initialize_position_embeddings(self):
        position_embeddings = torch.zeros(self.block_size, self.n_embd)
        for pos in range(self.block_size):
            for i in range(0, self.n_embd, 2):
                position = torch.tensor(pos, dtype=torch.float)
                exponent = torch.tensor(i / self.n_embd, dtype=torch.float)
                position_embeddings[pos, i] = torch.sin(position / (10000 ** exponent))
                position_embeddings[pos, i + 1] = torch.cos(position / (10000 ** exponent))
        return position_embeddings

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