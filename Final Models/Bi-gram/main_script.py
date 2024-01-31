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
max_iters = params['max_iters']
eval_interval = params['eval_interval']
eval_iters = params['eval_iters']
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

from tokenizer import EncoderDecoder
encoder_decoder = EncoderDecoder()
encoder_decoder.train_tokenizer(captions, vocab_size=5000)

input_data = encoder_decoder.encode(corpus)
print("total tokens", len(input_data))

# train-test split
n = int(0.8*len(input_data))
train_data = input_data[:n]
val_data = input_data[n:]

train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)
vocab_size = len(encoder_decoder.tokenizer.get_vocab())

decoded_text = encoder_decoder.decode(train_data[:20].tolist())
print(f"train data {train_data[:20]}")
print(f"decoded data {decoded_text}")

from bigram_model import BigramLanguageModel
from torch.nn import DataParallel

model = BigramLanguageModel(n_embd, block_size, dropout, n_head, n_layer, vocab_size)
model = DataParallel(model)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 

print(sum(p.numel() for p in m.parameters())/1e6, 'Million parameters')

from train_bigram import train_model
steps, train_losses, val_losses = train_model(model, optimizer, max_iters, eval_interval, eval_iters, train_data, val_data, block_size, batch_size, device)

print(f"train losses {train_losses[:5]}\ntest losses {val_losses[:5]}")

import matplotlib.pyplot as plt

plt.plot(range(steps), train_losses, label='Train Loss')
plt.plot(range(0, steps, eval_interval), val_losses, label='Validation Loss', marker='o', linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

end_time = timeit.default_timer()
print(f"model trained in {(end_time-start_time) / 60 }mins")

# context = 'Let me tell you a story of'
# input_tokens = torch.tensor(encoder_decoder.encode(context), dtype=torch.long)
# print(input_tokens)
# generated_output = BigramLanguageModel.generate(idx=input_data, max_new_tokens=20)[0]

# print(f"generated output:")
# print(f"'{context}' {encoder_decoder.decode(generated_output)}")