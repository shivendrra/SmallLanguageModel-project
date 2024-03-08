"""
  -> this is a proper seq-2-seq transformer model with encoder and decoder layers
  -> this is a more complex one, just for understanding
"""

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

class FeedForward:
  def __init__(self, n_embd):
    super(FeedForward, self).__init__()
    self.fc1 = nn.Linear(n_embd, 4*n_embd)  ## n_ff = 4*n_embd
    self.fc2 = nn.Linear(4*n_embd, n_embd)
  
  def forward(self, x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, n_embd, n_head, n_ff, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attention = MultiHeadAttention(n_embd, n_head)
    self.feed_forward = FeedForward(n_embd, n_ff)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(n_embd)
    self.norm2 = nn.LayerNorm(n_embd)
  
  def forward(self, src, src_mask):
    src2 = self.self_attention(src, src, src, src_mask)
    src = src + self.dropout(src2)
    src = self.norm1(src)

    src2 = self.feed_forward(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)
    
    return src

class DecoderLayer(nn.Module):
  def __init__(self, n_embd, n_head, n_ff, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attention = MultiHeadAttention(n_embd, n_head)
    self.encoder_attention = MultiHeadAttention(n_embd, n_head)
    self.feed_forward = FeedForward(n_embd, n_ff)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(n_embd)
    self.norm2 = nn.LayerNorm(n_embd)
    self.norm3 = nn.LayerNorm(n_embd)
  
  def forward(self, trg, enc_src, trg_mask, src_mask):
    trg2 = self.self_attention(trg, trg, trg, trg_mask)
    trg = trg + self.dropout(trg2)
    trg = self.norm1(trg)

    trg2 = self.encoder_attention(trg, enc_src, enc_src, src_mask)
    trg = trg + self.dropout(trg2)
    trg = self.norm2(trg)

    trg2 = self.feed_forward(trg)
    trg = trg + self.dropout(trg2)
    trg = self.norm3(trg)

    return trg

class TransformerModel(nn.Module):
  def __init__(self, n_embd, n_ff, n_head, n_layers, dropout, vocab_size, block_size, batch_size, src_vocab_size, trg_vocab_size,):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.encoder_layers = nn.ModuleList([EncoderLayer(n_embd, n_head, n_ff, dropout) for _ in range(n_layers)])
    self.decoder_layers = nn.ModuleList([DecoderLayer(n_embd, n_head, n_ff, dropout) for _ in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.fc_out = nn.Linear(n_embd, vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)
    self.n_embd = n_embd

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

    tok_embd = self.token_embedding_table(idx)
    pos_embd = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_embd + pos_embd

    for layer in self.encoder_layers:
      x = layer(x, None)
        
    for layer in self.decoder_layers:
      x = layer(x, x)
    
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    
    else:
      B, T, C = logits.shape()
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