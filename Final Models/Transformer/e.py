import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1400)
class Head(nn.Module):
  """ one head of self attention """

  def __init__(self, n_embd, n_head, dropout, block_size):
    head_size = n_embd // n_head
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
  
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * (C // self.tril.size(-1))**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
  
    # perform the weighted aggregation of the values
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  """ Multiple heads of self-attention in parallel"""

  def __init__(self, n_embd, n_head, dropout, block_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(n_embd, n_head, dropout, block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * (n_embd // n_head), n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(out)
    
    return out

class FeedForward(nn.Module):
  """ simple linear layer followed by non-linearity """

  def __init__(self, n_embd, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.GELU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head, dropout, block_size):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size, dropout, block_size)
    self.ffwd = FeedForward(n_embd, dropout)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class TransformerModel(nn.Module):
  def __init__(self, n_embd, block_size, dropout, n_head, n_layer, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding) and module.weight.numel() > 0:
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    print("Shape of x before lm_head:", x.shape)
    logits = self.lm_head(x) # (B,T,vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets, ignore_index=-52, reduction='mean')
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :] # becomes (B, C)
      probs = F.softmax(logits, dim=-1) # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    return idx, loss