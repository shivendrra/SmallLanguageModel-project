"""
  -> main bi-gram model, from Karpathy's lecture
  -> exact same but with a sub-word level tokenizer, hence better
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  """ one head of self attention """
  
  def __init__(self, n_embd, dropout, block_size, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    weight = q @ k.transpose(-2, -1) * C**-0.5
    weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
    weight = F.softmax(weight, dim=-1)
    weight = self.dropout(weight)
    v = self.value(x)
    output = weight @ v
    return output
  
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """
  def __init__(self, n_embd, n_head, dropout, block_size, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(n_embd, dropout, block_size, head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    output = torch.cat([h(x) for h in self.heads], dim=-1)
    output = self.dropout(self.proj(output))

    return output

class FeedForward(nn.Module):
  """ a simple linear layer followed by non-linearity """
  def __init__(self, n_embd, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(n_embd, 4*n_embd),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computatuion """
  
  def __init__(self, n_embd, n_head, dropout, block_size):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_embd, n_head, dropout, block_size, head_size)
    self.ffwd = FeedForward(n_embd, dropout)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramModel(nn.Module):
  
  def __init__(self, n_embd, block_size, dropout, n_head, n_layer, vocab_size, device):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.device = device
    self.block_size = block_size

  def forward(self, idx, targets=None):
    B, T = idx.shape

    token_embd = self.token_embedding_table(idx)
    position_embd = self.position_embedding_table(torch.arange(T, device=self.device))
    x = token_embd + position_embd
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-self.block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:,-1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx