import torch
import torch.nn as nn
from torch.nn import functional as F

# setting the hyperparameters
batch_size = 32  # no of independent sequences to be ran in parallel
block_size = 8  # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

# reading the text data for training
import os

os.chdir("D:/Machine Learning/SLM-Project/")
with open('Data/training_data.txt', 'r', encoding='utf-8') as file:
  text = file.read()

# sorting all the unique caharcters in the data
chars = sorted(list(set(text)))
vocab_size = len(chars)

#  encoder and decoder of the text
string_to_index = { ch:i for i,ch in enumerate(chars) }
index_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string_to_index[c] for c in s]
decode = lambda l: ''.join([index_to_string[i] for i in l])

# train-test spliting
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train and last 10% will be test
train_data = data[:n]
test_data = data[n:]

# data loading
def get_batch(split):
  # generate a small batch of data of inputs of x and y
  data = train_data if split == 'train' else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'test']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# simple bigram model
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()

    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, index, targets=None):

    B, T = index.shape
    
    token_embd = self.token_embedding_table(index) # B, T, C
    pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # T, C
    x = token_embd + pos_embd # B, T, C
    logits = self.lm_head(token_embd) # B, T, vocab_size

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, index, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, _ = self(index, None) # get the predictions
      logits = logits[: ,-1, :] # focus on the last step // becomes (B, C)

      probs = F.softmax(logits, dim=-1) # applying softmax to get the probabilities // (B, C)
      index_next = torch.multinomial(probs, num_samples=1) # sample from the distribution // (B, 1)

      index = torch.cat((index, index_next), dim=1) # append sampled indexes to the running sequence // (B, T+1)

    return index

model = BigramLanguageModel()
m = model.to(device)

# creating an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  
  # evaluate losses on every step in training and test data
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

  # get the sample of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_output = decode(m.generate(context, max_new_tokens=500)[0].tolist())

print(generated_output)