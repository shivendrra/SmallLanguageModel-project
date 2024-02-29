import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head:
  pass

class MultiAttentionHead:
  pass

class FeedForward:
  def __init__(self, n_embd, n_ff):
    super(FeedForward, self).__init__()
    self.fc1 = nn.Linear(n_embd, n_ff)
    self.fc2 = nn.Linear(n_ff, n_embd)
  
  def forward(self, x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, n_embd, n_head, n_ff, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attention = MultiAttentionHead(n_embd, n_head)
    self.encoder_attention = MultiAttentionHead(n_embd, n_head)
    self.feed_forward = FeedForward(n_embd, n_ff)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(n_embd)
    self.norm2 = nn.LayerNorm(n_embd)
    self.norm3 = nn.LayerNorm(n_embd)
  
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
    self.self_attention = MultiAttentionHead(n_embd, n_head)
    self.encoder_attention = MultiAttentionHead(n_embd, n_head)
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
  def __init__(self, n_embd, n_ff, n_head, n_layers, dropout, vocab_size, block_size, batch_size):
    super().__init__()

    self.token_embeddings = nn.Embedding(vocab_size, n_embd)
    self.positional_embeddings = nn.Embedding(block_size, n_embd)
    self.encoder_layers = nn.ModuleList([EncoderLayer(n_embd, n_head, n_ff, dropout) for _ in range(n_layers)])
    self.decoder_layers = nn.ModuleList([DecoderLayer(n_embd, n_head, n_ff, dropout) for _ in range(n_layers)])
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

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_embd = self.token_embeddings(idx)
    pos_embd = self.positional_embeddings(torch.arange(T, device=device))
    x = tok_embd + pos_embd

  def generate(self, idx, max_tokens=50):
    for _ in range(max_tokens):
      idx_cond = idx[:, -self.block_size: ]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx, loss