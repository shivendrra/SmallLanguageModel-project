import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class BiGramAttention(nn.Module):
    """Bi-gram attention mechanism"""

    def __init__(self, n_embd, dropout, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Consider only the last two tokens
        k = k[:, -2:, :]
        q = q[:, -2:, :]

        # Compute attention scores
        weight = q @ k.transpose(-2, -1) * C ** -0.5  # (B, 2, 2)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        
        # Weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        v = v[:, -2:, :]
        output = weight @ v  # (B, 2, head_size)
        return output

class BiGramBlock(nn.Module):
    """Bi-gram transformer block"""

    def __init__(self, n_embd, dropout, head_size):
        super().__init__()
        self.bi_gram_attention = BiGramAttention(n_embd, dropout, head_size)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.bi_gram_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class BiGramTransformer(nn.Module):
    """Bi-gram transformer model"""

    def __init__(self, n_embd, dropout, n_head, n_layer, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[BiGramBlock(n_embd, dropout, n_embd // n_head) for _ in range(n_layer)])
        self.layer_norm_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.vocab_size = vocab_size
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)
        
        # Consider only the last two tokens
        token_embd = token_embd[:, -2:, :]
        x = self.blocks(token_embd)
        
        # Reshape the output tensor for linear layer
        x = x[:, -1, :]  # Consider only the last token
        x = self.layer_norm_final(x)  # Apply layer normalization
        
        logits = self.lm_head(x)  # Final linear layer
        
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
