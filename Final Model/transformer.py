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