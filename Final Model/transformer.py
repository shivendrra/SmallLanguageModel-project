import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64    # indendent sequences in parallel
block_size = 128    # maximum context length
max_iters = 5000    # max epochs
eval_interval = 1000    # output iterations
learning_rate = 3e-4    # learning rate for the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'    # device to run calculations on
eval_iters = 200    # iterations to perform evaluations
n_embd = 8    # embeddings
n_head = 8    # self attention heads in parallel
n_layer = 4    # layers in deep net
dropout = 0.2    # dropout rate

torch.manual_seed(1400)

# import the data
import os
os.chdir('D:/Machine Learning/SLM-Project/')
with open('Data/training_data.txt', 'r', encoding='utf-8') as file:
  text = file.read()

# list of all unique characters present in data
chars = sorted(list(set(text)))
vocab_size = len(chars)

# simple character level tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading in batches
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])   # generates a batch of tokens of block_size
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # generates the batch of tokens block_size with 1 next element and first element is popped out
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()   # helps to reduce memory consumption and speed up computation by removing gradients during evolution process
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.mean()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  """ one head of self attention """

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)   # (B,T,C)
    q = self.query(x)   # (B,T,C)

    # computing affinities
    weight = q @ k.transpose(-2, -1) * C**-0.5    # query matrix and key matrix dot product and multiplying dimensions' square root
    weight = weight.masked_fill(self.trill[:T, :T] == 0, float('-inf'))   # masking triangular part of the matrix 
    weight = F.softmax(weight, dim=1)   # applying softmax for better probabilities
    weight = self.dropout(weight)   # droping out some parts

    # performing weitghted aggregation of the values
    v = self.value(x)   # (B,T,C
    out = weight @ v   # dot product of weights and values

    return out

class MultiHeadAttention(nn.Module):
  """ Multiple heads of self-attention in parallel"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(out)
    
    return out

class FeedForward(nn.Module):
  """ simple linear layer followed by non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class TransformerModel(nn.Module):
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

model = TransformerModel(vocab_size)
model = model.to(device)
n_param = sum(p.numel() for p in model.parameters()) / 1e6
print(f"no of parameters present are {n_param} million")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

  # every once in a while evaluate the loss on train and val sets
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  #  sample a batch from the data
  xb, yb = get_batch('train')

  # loss evaluation
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# save the trained model
torch.save(model.state_dict(), f"{n_param:.1f}_transformer_model.pth")