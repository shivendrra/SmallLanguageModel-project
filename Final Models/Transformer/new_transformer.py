import torch
import numpy as np

class nnMods:
    """ Neural Network Modules """
    class TokenEmbeddings:
        def __init__(self, vocab_size, n_embd):
            self.vocab_size = vocab_size
            self.n_embd = n_embd
            self.token_embeddings = self.forward()

        def forward(self):
            token_embeddings = torch.randn(vocab_size, n_embd)
            token_embeddings /= torch.norm(token_embeddings, dim=1, keepdim=True)
            return token_embeddings

    class PositionalEmbeddings:
        def __init__(self, block_size, n_embd):
            self.block_size = block_size
            self.n_embd = n_embd
            self.positional_embeddings = self.forward()

        def forward(self):
            position_embeddings = torch.zeros(block_size, n_embd)
            for pos in range(block_size):
                for i in range(0, n_embd, 2):
                    position = torch.tensor(pos, dtype=torch.float)
                    exponent = torch.tensor(i / n_embd, dtype=torch.float)
                    position_embeddings[pos, i] = torch.sin(position / (1e4 ** exponent))
                    position_embeddings[pos, i+1] = torch.cos(position / (1e4 ** exponent))
            return position_embeddings

    class Sequential:
        @staticmethod
        def process(*modules):
            for module in modules:
                x = module(x)
            return x

    class LinearLayer:
        def __init__(self, input_dim, output_dim):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.weights = torch.randn(output_dim, input_dim)
            self.bias = torch.zeros(output)
        
        def forward(self, x):
            output = torch.matmul(x, self.weights.t()) + self.bias
            return output

    class LayerNorm:
        def __init__(self, n_features, eps=1e-5):
            self.n_features = n_features
            self.eps = eps
            self.gamma = torch.ones(n_features)
            self.beta = torch.zeros(n_features)

        def forward(self, x):
            # mean and standard deviation along the feature dimension
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            
            # normalization
            x_normalized = (x - mean) / (std + self.eps)
            
            # scale and shift
            y = self.gamma * x_normalized + self.beta
            return y

class MultiHeadAttention:
    pass

class FeedForward:
    pass

class Block(nnMods):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size, dropout, block_size)
        self.feedForward = FeedForward(n_embd, dropout)
        self.linear1 = nnMods.LayerNorm(n_embd)
        self.linear2 = nnMods.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.attention(self.linear1.forward(x))
        x = x + self.feedForward(self.linear2.forward(x))
        return x

class CustomTransformerModel(nnMods):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nnMods.TokenEmbeddings(vocab_size, n_embd).forward()
        self.position_embedding_table = nnMods.PositionalEmbeddings(block_size, n_embd).forward()
        self.blocks = nnMods.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.linear_final = nnMods.LayerNorm(n_embd)
        self.lm_head = nnMods.LinearLayer(n_embd, vocab_size)

    def forward(self, idx):
        tok_emb = self.token_embedding_table[idx]  # (B, T, C)
        pos_emb = self.position_embedding_table[:tok_emb.shape[1]]  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        return x

batch_size = 64
vocab_size = 483
block_size = 128
n_embd = 8
n_head = 4
n_layer = 4
dropout = 0.1
idx = torch.randint(0, vocab_size, (batch_size, block_size))

model = CustomTransformerModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout)
token_embeddings = model.token_embedding_table
position_embeddings = model.position_embedding_table
output = model.forward(idx)

print("inputs: ", idx.shape)
print("Token Embeddings Shape:", token_embeddings.shape)
print("Positional Embeddings Shape:", position_embeddings.shape)
print("Output Shape:", output.shape)