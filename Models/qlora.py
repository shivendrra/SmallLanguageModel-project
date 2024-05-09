"""
  use this file to implement LoRA fintuning on the base model that you trained
  -> just needs a fine-tuning dataset to be loaded
  -> run it as it is
  -> Not Tested
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QLoRA(nn.Module):
    def __init__(self, model, alpha=0.01, bits=8):
        super(QLoRA, self).__init__()
        self.model = model
        self.alpha = alpha
        self.bits = bits

        self.scaling_factors = nn.Parameter(torch.ones(len(model.layers)))
        self.quantization_levels = 2 ** bits - 1  # Number of quantization levels

    def forward(self, x):
        return self.model(x)

    def finetune(self, x, y):
        activations = [x]
        for layer in self.model.layers:
            x = layer(x)
            activations.append(x)

        original_loss = self.model.compute_loss(activations[-2], y)

        adjusted_loss = self.model.compute_loss(self.adjust_activations(activations), y)

        gradients = torch.autograd.grad(outputs=adjusted_loss, inputs=self.scaling_factors,
                                        grad_outputs=torch.ones_like(adjusted_loss),
                                        create_graph=True, retain_graph=True)

        with torch.no_grad():
            for i, layer_sf in enumerate(self.scaling_factors):
                layer_sf.grad = gradients[i]
                layer_sf.data = layer_sf.data - self.alpha * layer_sf.grad

    def adjust_activations(self, activations):
        adjusted_activations = []
        for i, activation in enumerate(activations[:-1]):
            scaling_factor = torch.sigmoid(self.scaling_factors[i])
            adjusted_activation = activation * scaling_factor
            # Quantization
            adjusted_activation = torch.round(adjusted_activation * self.quantization_levels) / self.quantization_levels
            adjusted_activations.append(adjusted_activation)
        adjusted_activations.append(activations[-1])
        return adjusted_activations

import json
import os
from tokenizer import Tokenizer
from decoder import GPT

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('../datasets/wiki_176m.txt', 'r', encoding='utf-8') as file:
  data = file.read()

print(f"{(len(data)/1e6):.2f} million letters")

tokenizer = Tokenizer()
vocab_size = tokenizer.get_vocab()

# Train and test splits
data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

with open('config.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

# Required parameters
batch_size = params['batch_size']
block_size = params['block_size']
max_iters = 1000
eval_interval = 100
eval_iters = 200
learning_rate = params['learning_rate']
torch.manual_seed(1400)


def get_batch(split):

  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out

model = GPT(vocab_size)
model = model.to(device)
lora = QLoRA(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
n_param = sum(p.numel() for p in model.parameters())/1e6
print(f"{n_param:.2f} million")

steps = []
train_losses = []
val_losses = []

for iter in range(max_iters):
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    steps.append(iter)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])

  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)

  # LoRA fine-tuning
  lora.finetune(xb, yb)

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

torch.save(model.state_dict(), f'GPT_{n_param:.0f}m.pth')