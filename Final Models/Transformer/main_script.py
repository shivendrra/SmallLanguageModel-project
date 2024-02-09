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

# importing training data for bpe
file_path = 'Data/training_data.txt'
with open(file_path, 'r', encoding='utf-8') as file:
  captions = file.read()

# importing training data for model
with open('Data/captions.txt', 'r', encoding='utf-8') as file:
  corpus = file.read()

# tokenizing the data
from tokenizer import EncoderDecoder
encoder_decoder = EncoderDecoder()
encoder_decoder.train_tokenizer(captions, vocab_size=10000)
input_data = encoder_decoder.encode(captions)

# train-test split
n = int(0.9*len(input_data))
train_data = input_data[:n]
val_data = input_data[n:]

train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)
vocab_size = len(encoder_decoder.tokenizer.get_vocab())

from transformer import TransformerModel
model = TransformerModel(n_embd, block_size, dropout, n_head, n_layer, vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
n_param = sum(p.numel() for p in model.parameters()) / 1e6

print(f"no of parameters present are {n_param} million")

from train_model import TrainModel
steps, train_loss, val_loss= TrainModel(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)

# saving the model
torch.save(model.state_dict(), f"{n_param:.1f}m_transformer_model.pth")

# generating output
context = 'I sometimes feel so ignored that'
max_tokens = 100
gen_input = torch.tensor(encoder_decoder.encode(context), dtype=torch.long)
gen_output, gen_loss = model.generate(gen_input, max_tokens)
gen_output = encoder_decoder.decode(gen_output)

print(f"generation loss: {gen_loss}")
print(f"generated output after '{context}' {gen_output} ")