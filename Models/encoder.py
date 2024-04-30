import json
with open('config.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

# required parameters
block_size = params['block_size']
d_model = params['d_model']
n_head = params['n_heads']
n_layers = params['n_layers']
learning_rate = params['learning_rate']
dropout = params['dropout']
norm_eps = params['norm_eps']

import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):
  """
    using RMS normalization instead of layer norm for better stability while training

  Args:
    dim (int): dimensions to apply normalizations
    eps (float): small floating point no to prevent division by zero
  """
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

class SingleHead(nn.Module):
  """
    single head of self attention

    generates three copies of 'x' -> Wk, Wq, Wv
      
      `attention matrix = '[Wq . Wk.transpose()]/dim(Wk)**-0.5'`
      if masking is True:
        [[9, 2, 5, 9, 6]        [[9, 0, 0, 0, 0]
         [6, 9, 2, 6, 2]         [6, 9, 0, 0, 0]
         [1, 7, 9, 6, 2]  -->    [1, 7, 9, 0, 0]
         [8, 3, 2, 9, 2]         [8, 3, 2, 9, 0]
         [2, 1, 2, 6, 9]]        [2, 1, 2, 6, 9]]
      
         `outputs = softmax(attention_matrix) . Wv`

  Args:
    head_size (int): size of each head for attention
    d_model (int): model size dimensions
    block_size (int): context size
    dropout (float): % of dropout connections
  """
  def __init__(self,
      head_size: int,
      d_model: int,
      block_size: int,
      dropout: float):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x: torch.Tensor, mask: bool= False):
    B, T, C = x.shape
    key = self.key(x)
    query = self.query(x)
    scores = torch.matmul(query ,key.transpose(-2, -1)) / (key.shape[-1]**-0.5)

    if mask is True:
      scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

    att_mat = F.softmax(scores, dim=-1)
    att_mat = self.dropout(att_mat)
    value = self.value(x)
    output = torch.matmul(att_mat, value)

    return output

class MultiHeadAttention(nn.Module):
  """
    multiple head of self-attention running in parallel
    outputs are then concatenated into a single matrix

  Args:
    d_model (int): model size dimensions
    block_size (int): context size
    n_head (int): no of attention head in parallel
    dropout (float): % of dropout connections
  """
  def __init__(self,
      d_model: int,
      block_size: int,
      n_head : int,
      dropout: float):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([SingleHead(d_model=d_model, dropout=dropout, block_size=block_size, head_size=head_size) for _ in range(n_head)])
    self.projection = nn.Linear(head_size * n_head, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, mask: bool):
    out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
    out = self.dropout(self.projection(out))

    del x
    return out

class FeedForward(nn.Module):
  """
    simple feedforward layer using two linear layers and LeakyRelu as activation function
    `LeakyRelu(x) = {x, if x >= 0 ; neg_slope*x, otherwise}`
    `LeakyReLU(x) = max(0,x) + negative_slope*min(0,x)`

    expansion factor = 5
  
  Args:
    d_model (int): model size dimensions
    dropout (float): % of dropout connections
  """
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 4 * d_model),
      nn.LeakyReLU(0.1),
      nn.Linear(4 * d_model, d_model),
      nn.Dropout(dropout),
      )

  def forward(self, x: torch.Tensor):
    return self.net(x)

class EncoderBlock(nn.Module):
  """
    simple encoder block with one layer of self-attention followed by feedforward
    but with additional layer of masked attention layer for next token prediction

  Args:
    block_size (int): context size
    n_head (int): no of attention head in parallel
    norm_eps (float): small floating point no to prevent division by zero
    dropout (float): % of dropout connections
  """
  def __init__(self, d_model: int,
        block_size: int,
        n_head: int,
        norm_eps: float,
        dropout: float):
    super().__init__()
    self.self_att = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, eps=norm_eps)

  def forward(self, x: torch.Tensor):
    x_out = self.self_att(self.norm(x), mask=False)
    x_out = x + self.dropout(x_out)

    x_out = self.ffwd(self.norm(x_out))
    x_out = x + self.dropout(x_out)
    
    # extra layer of masked attention for next-token prediction
    x = self.self_att(self.norm(x_out), mask=True)
    x = x_out + self.dropout(x)

    del x_out
    return x

class BERT(nn.Module):
  def __init__(self, vocab_size: int):
    super().__init__()
    self.block_size = block_size
    self.tok_embed = nn.Embedding(vocab_size, d_model)
    self.pos_embed = nn.Embedding(block_size, d_model)
    self.decoder = nn.Sequential(*[EncoderBlock(n_head=n_head, d_model=d_model, dropout=dropout, norm_eps=norm_eps, block_size=block_size) for _ in range(n_layers)])
    self.norm_final = RMSNorm(d_model, eps=norm_eps)
    self.linear_final = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    toks = self.tok_embed(idx)
    pos = self.pos_embed(torch.arange(T, device=device))
    x = toks + pos

    x = self.decoder(x)
    logits = self.linear_final(self.norm_final(x))

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
      generate new tokens using the trained model

    Args:
      - idx (Tensor): input tensor representing initial token indices
      - max_new_tokens (int): max no of new tokens to generate
      - temperature (float): softmax temperature for sampling
      - top_k (int): no of top tokens to consider in sampling

    Returns:
      - generated_tokens (list): list of generated token indices
    """
    self.eval()
    for _ in range(max_new_tokens):

      idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] / temperature

      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx