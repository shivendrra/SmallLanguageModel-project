import json
import os
os.chdir('D:/Machine Learning/SLM-Project')

with open('Final Models/Bi-gram/hyperparams.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

# importing training data
file_path = 'Data/captions.txt'
with open(file_path, 'r', encoding='utf-8') as file:
  data = file.read()
chars = sorted(list(set(data)))
vocab_size = len(chars)

print(f"list of unique characters in dataset{''.join(chars)}")
print(f"vocab size is {vocab_size}")

import torch
from encoder import EncoderDecoder
ed = EncoderDecoder(n_iters=20, input_data=data)

data = ed.encoder()
input_data = torch.tensor(data, dtype=torch.long)

# train-test split
n = int(0.9*len(input_data))
train_data = data[:n]
val_data = data[n:]

print(train_data[:20])

batch_size = params['batch_size']
block_size = params['block_size']
max_iters = params['max_iters']
eval_interval = params['eval_intervals']
eval_iters = params['eval_iters']
n_head = params['n_head']
n_embd = params['n_embd']
n_layer = params['n_layer']
dropout = params['dropout']
learning_rate = params['learning_rate']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from bigram_model import BigramLanguageModel

model = BigramLanguageModel(n_embd, block_size, dropout, n_head, n_layer, vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
print(sum(p.numel() for p in m.parameters())/1e6, 'Million parameters')

from train_bigram import train_model
train_model(model, optimizer, max_iters, eval_interval, eval_iters, train_data, val_data, block_size, batch_size, device)