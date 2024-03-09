import numpy as np
import pandas as pd

# reading file for training
with open('Data/captions.txt', 'r', encoding='utf-8') as file:
  dataset = file.read()

# implementing sub-word level tokenizer
from encoder import EncoderDecoder

tokenize = EncoderDecoder(dataset, n_iters=20)
token_inputs = np.array(tokenize.encoder(dataset))

n = int(0.8*len(token_inputs))
train_data = token_inputs[:n]
val_data = token_inputs[n:]

print(train_data[:20], val_data[:20])
print(tokenize.decoder(train_data)[:20], tokenize.decoder(val_data)[:20])

# implementing rnn
from mainRNN import SimpleRNN

input_size = 1
hidden_size = 2
output_size = 1
rnn = SimpleRNN(input_size, hidden_size, output_size)

epochs = 10
learning_rate = 0.1

for epoch in range(epochs):
  total_loss = 0
  for i in range(len(train_data)):
    x = np.array(train_data[:i + 1])
    y = np.array([train_data[:i + 1]])

    # Forward and backward pass
    loss = rnn.backward(x, y, learning_rate)
    total_loss += loss

    # if epoch % 100 == 0:
    print(f'Epoch {epoch}, Loss: {total_loss}')