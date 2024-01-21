import json
import os
os.chdir('D:/Machine Learning/SLM-Project')

with open('Final Models/Bi-gram/hyperparams.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

import torch

batch_size = params['batch_size']
block_size = params['block_size']
max_iters = params['max_iters']
eval_interval = params['eval_interval']
eval_iters = params['eval_iters']
n_head = params['n_head']
n_embd = params['n_embd']
n_layer = params['n_layer']
dropout = params['dropout']
learning_rate = params['learning_rate']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# importing training data
file_path = 'Data/captions.txt'
with open(file_path, 'r', encoding='utf-8') as file:
  captions = file.read()

chars = sorted(list(set(captions)))
vocab_size = len(chars)

print(f"list of unique characters in dataset: {''.join(chars)}")
print(f"vocab size is {vocab_size}")

from encoder import EncoderDecoder
ed = EncoderDecoder(n_iters=50, train_data=captions)

input_data = ed.encoder(captions)

# train-test split
n = int(0.9*len(input_data))
train_data = input_data[:n]
val_data = input_data[n:]

train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)

# print(train_data[30:105])
# print(val_data[:20])

# print(ed.decoder(train_data[30:105]))
# print(ed.decoder(val_data[:20]))

from bigram_model import BigramLanguageModel

model = BigramLanguageModel(n_embd, block_size, dropout, n_head, n_layer, vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
print(sum(p.numel() for p in m.parameters())/1e6, 'Million parameters')

from train_bigram import train_model
iter, losses = train_model(m, optimizer, max_iters, eval_interval, eval_iters, train_data, val_data, block_size, batch_size, device)

print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")