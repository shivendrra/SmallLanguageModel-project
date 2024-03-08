import json
import os
os.chdir('D:/Machine Learning/SLM-Project')

with open('Final Models/Transformer/hyperparams.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

import torch

batch_size = params['batch_size']
block_size = params['block_size']
n_head = params['n_head']
n_embd = params['n_embd']
n_layer = params['n_layer']
dropout = params['dropout']
learning_rate = params['learning_rate']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# importing training data for model
file_path = 'Data/training_data.txt'
with open(file_path, 'r', encoding='utf-8') as file:
  captions = file.read()

# tokenizing the data
from tokenizer import Tokenizer
tokenizer = Tokenizer()
input_data = tokenizer.encode(captions)

# train-test split
n = int(0.9*len(input_data))
train_data = input_data[:n]
val_data = input_data[n:]

train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)
vocab_size = tokenizer.get_vocab()

from transformer import TransformerModel
model = TransformerModel(n_embd=n_embd, block_size=block_size, dropout=dropout, n_head=n_head, n_layer=n_layer, vocab_size=vocab_size, norm_eps=1e-5)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
n_param = sum(p.numel() for p in model.parameters()) / 1e6

print(f"no of parameters present are {n_param} million")

from train_model import TrainModel
train = TrainModel(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)
train.train_model()

# saving the model
torch.save(model.state_dict(), f"{n_param:.1f}m_model.pth")

# generating output
target_text = "Would you like to tell me your name because "
context = torch.tensor([tokenizer.encode(target_text)], dtype=torch.long, device=device)
generated_output = tokenizer.decode(model.generate(context, max_new_tokens=10)[0].tolist())
print(generated_output)