import json
import os
os.chdir('D:/Machine Learning/SLM-Project')

with open('Final Models/Bi-gram/hyperparams.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

import torch
import timeit

start_time = timeit.default_timer()

batch_size = params['batch_size']
block_size = params['block_size']
n_head = params['n_head']
n_embd = params['n_embd']
n_layer = params['n_layer']
dropout = params['dropout']
learning_rate = params['learning_rate']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# importing training data for model
with open('Data/captions.txt', 'r', encoding='utf-8') as file:
  corpus = file.read()
  print(len(corpus)/1e6, "million words")

from tokenizer import Tokenizer
encoder_decoder = Tokenizer()
input_data = encoder_decoder.encode(corpus)
print("total tokens", len(input_data)/1e3, 'thousand')

# train-test split
n = int(0.8*len(input_data))
train_data = input_data[:n]
val_data = input_data[n:]

train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)
vocab_size = encoder_decoder.get_vocab()

from bigram_model import BiGramTransformer

model = BiGramTransformer(n_embd, dropout, n_head, n_layer, vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
n_params = sum(p.numel() for p in m.parameters())/1e6
print(n_params, 'Million parameters')

from train_model import TrainBiGramModel
trainer = TrainBiGramModel(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)
trainer.train_model()

end_time = timeit.default_timer()
print(f"model trained in {(end_time-start_time) / 60 }mins")

context = 'Let me tell you a story of'
input_tokens = torch.tensor(encoder_decoder.encode(context), dtype=torch.long, device=device)
generated_output = m.generate(idx=input_data, max_new_tokens=20).tolist()

print(f"generated output:")
print(f"'{context}' {encoder_decoder.decode(generated_output)}")