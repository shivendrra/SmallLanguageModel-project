import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1400)

class Head(nn.Module):
  def __init__(self, d_embd, n_head, dropout, block_size):
    head_size = d_embd // n_head
    super().__init__()
    self.key = nn.Linear(d_embd, head_size, bias=True)
    self.query = nn.Linear(d_embd, head_size, bias=True)
    self.value = nn.Linear(d_embd, head_size, bias=True)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    pass

class MultiHeadAttention(nn.Module):
  def __init__(self, d_embd, n_head, dropout, block_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(d_embd=d_embd, n_head=n_head, dropout=dropout, block_size=block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * (d_embd // n_head), d_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(out)
    
    return out

class FeedForward:
  def __init__(self, d_embd):
    super().__init__()
    self.fc1 = nn.Linear(d_embd, 4*d_embd)  # n_ff = 4*d_embd
    self.fc2 = nn.Linear(4*d_embd, d_embd)
  
  def forward(self, x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x
  
class EncoderDecoderAttention(nn.Module):
  def __init__(self, d_embd, n_head, dropout, block_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(d_embd, n_head, dropout, block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * (d_embd // n_head), d_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, key, value, mask=None):
    x = torch.cat((key, query, value), dim=-1)
    energies = []
    for head in self.heads:
        energy = head(x)
        energies.append(energy.unsqueeze(1))
    energy = torch.cat(energies, dim=1)
    energy = self.proj(energy)
    energy = self.dropout(energy)

    if mask is not None:
      energy = energy.masked_fill(mask == 0, float('-inf'))

    attention = F.softmax(energy, dim=-1)
    output = torch.matmul(attention, value)

    return output

class EncoderLayer(nn.Module):
  def __init__(self, d_embd, n_head, dropout, block_size):
    super().__init__()
    self.s_att = MultiHeadAttention(d_embd=d_embd, n_head=n_head, block_size=block_size, dropout=dropout)
    self.ffwd = FeedForward(d_embd=d_embd)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_embd)
    self.norm2 = nn.LayerNorm(d_embd)
  
  def forward(self, src, src_mask=None):
    src2 = self.s_att(src)
    src = src + self.dropout(src2)
    src = self.norm1(src)

    src2 = self.ffwd(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)
  
    return src

class DecoderLayer(nn.Module):
  def __init__(self, d_embd, n_head, dropout, block_size) -> None:
    super().__init__()
    self.s_att = MultiHeadAttention(d_embd=d_embd, n_head=n_head, block_size=block_size, dropout=dropout)
    self.enc_att = EncoderDecoderAttention(d_embd=d_embd, n_head=n_head, block_size=block_size, dropout=dropout)
    self.ffwd = FeedForward(d_embd=d_embd)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_embd)
    self.norm2 = nn.LayerNorm(d_embd)
    self.norm3 = nn.LayerNorm(d_embd)
  
  def forward(self, trg, enc_src, trg_mask=None, src_mask=None):
    trg2 = self.s_att(trg)
    trg = trg2 + self.dropout(trg2)
    trg = self.norm1(trg)

    trg2 = self.enc_att(trg, enc_src, enc_src)
    trg = trg + self.dropout(trg2)
    trg = self.norm2(trg)

    trg2 = self.ffwd(trg)
    trg = trg + self.dropout(trg2)
    trg = self.norm3(trg)
  
    return trg

class Transformer(nn.Module):
  def __init__(self, block_size, vocab_size, n_layers, d_embd, n_head, dropout):
    super().__init__()
    self.d_embd = d_embd
    self.block_size = block_size
    
    self.token_embd = nn.Embedding(vocab_size, d_embd)
    self.pos_embd = nn.Embedding(block_size, d_embd)
    self.enc_layer = nn.ModuleList([EncoderLayer(n_head=n_head, block_size=block_size, dropout=dropout, d_embd=d_embd) for _ in range(n_layers)])
    self.dec_layer = nn.ModuleList([DecoderLayer(n_head=n_head, block_size=block_size, dropout=dropout, d_embd=d_embd) for _ in range(n_layers)])
    
    self.norm_final = nn.LayerNorm(d_embd)
    self.lm_head = nn.Linear(d_embd, vocab_size)
    self.fc_out = nn.Linear(d_embd, vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding) and module.weight.numel() > 0:
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def make_src_mask(self, src):
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask
    
  def make_trg_mask(self, trg):
    trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_embd = self.token_embd(idx)
    pos_embd = self.pos_embd(torch.arange(T, device=device))
    x = tok_embd + pos_embd

    for layer in self.enc_layer:
      x = layer(x, None)
        
    for layer in self.dec_layer:
      x = layer(x, x)
    
    x = self.norm_final(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets, ignore_index=-52, reduction='mean')
    
    return logits, loss
  
  def generate(self, idx, max_tokens=50):
    for _ in range(max_tokens):
      idx_cond = idx[:, -self.block_size: ]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx, loss