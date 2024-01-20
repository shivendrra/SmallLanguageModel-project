import os
os.chdir('D:/Machine Learning/SLM-Project/')

import numpy as np
import pandas as pd

# Define sigmoid activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
  return x * (1 - x)

class SimpleRNN:
  def __init__(self, input_size, hidden_size, output_size):
    
    # initializing weights and biases
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.w_ih = np.random.randn(hidden_size, input_size)
    self.w_hh = np.random.randn(hidden_size, hidden_size)
    self.w_oh = np.random.randn(output_size, hidden_size)
    
    self.b_h = np.zeros((hidden_size, 1))
    self.b_o = np.zeros((output_size, 1))
    
    # initializing the hidden state
    self.h = np.zeros((hidden_size, 1))


  def forward(self, x_seq):
    # forward pass through RNN
    outputs = []
    for x in x_seq:
      x = x.reshape(-1, 1)
      self.h = sigmoid(np.dot(self.w_ih, x) + np.dot(self.w_hh, self.h) + self.b_h)
      output = sigmoid(np.dot(self.w_oh, self.h) + self.b_o)

      outputs.append(output)
    return outputs
  
  def backward(self, x_seq, y_seq, learning_rate):
    # backward pass through RNN
    total_err = 0
    for t in range(len(x_seq)):
      x = x_seq[t]
      y = y_seq[t]

      # forward pass
      output = self.forward([x])[0]

      # backward pass
      error = y - output
      total_err += np.mean((error) **2)

      d_output = error * sigmoid_derivative(output)
      d_w_oh = np.dot(d_output, self.h.T)
      d_b_o = d_output

      # hidden layer gradients
      d_h = np.dot(self.w_ho.T, d_output) * sigmoid_derivative(self.h)
      d_w_ih = np.dot(d_h, x.T)
      d_w_hh = np.dot(d_h, self.h.T)
      d_b_h = d_h

      self.w_ih += learning_rate * d_w_ih
      self.w_hh += learning_rate * d_w_hh
      self.w_oh += learning_rate * d_w_oh
      self.b_h += learning_rate * d_b_h
      self.b_o += learning_rate * d_b_o
    
    return total_err

# # reading file for training
# with open('Data/captions.txt', 'r', encoding='utf-8') as file:
#   dataset = file.read()

dataset = """
- Listen up. Demato might not think you're behind this, but make no mistake, Judge Carmichael, I know a sophisticated
scam when I stop one. - Bravo. - So good. So good. Yeah, riveting. - Christina's performance
there is just, her ferocity. - Fantastic. - Welcome to a sneak peek at this season's most extraordinary and
heart pounding new show, "S.A.F.E Squad" starring
Christina Ricci, and also me, and others. Hi, I'm Steven Wu. - And I'm Betsy Chate. Our show follows a trio of investigators dedicated to stopping
criminals in their tracks. Now, imposter scams can
involve anyone who's pretending to be someone else in order to trick you into providing personal information or even sending money. - They're real and they're
not going away anytime soon. - We have so much to show you, so let's dive into a few
clips from our pilot episode of "S.A.F.E. Squad". (tense music) - Uh huh. Uh huh. So the message came from
the job posting site. Listen, why don't you come in today? We'll take a look, see what we can do. All right, Jordan. See you soon. Call the boss. Ask her to come in. - I know it's only my first day, but it sounds like a pretty
cut and dry job scam. Can't you and I just help this guy? - It's not that simple, rookie. He's the boss's baby brother. - Okay, baby bro. Walk
me through what happened. - So I'm on this job listing site and I get a message from a recruiter for a small shipping company. Small shipping company. It said I would be perfect for a role in their operations department. Company was called Travel in 88. Travel in 88. We specialize in your
unique piano shipping needs. The recruiter said, all I
needed to do was send $500 to cover mandatory software
training and job was mine. - Baby bro, come on. - If this website is a front, I mean, it's a pretty good one. - "I wouldn't trust anyone
else with my orwolu upright." - Ormolu. Have we compared the website's URL with the domain name of the
recruiter's email address? - They don't match up. I even tried the hotline
number at the bottom. Dead end.
"""

# implementing sub-word level tokenizer
from encoder import EncoderDecoder

tokenize = EncoderDecoder(dataset, n_iters=20)
token_inputs = np.array(tokenize.encoder())

input_size = token_inputs.shape[0]
hidden_size = 4
output_size = 1

rnn = SimpleRNN(input_size, hidden_size, output_size)

print(token_inputs.shape)
# Training loop
epochs = 10
learning_rate = 0.1

for epoch in range(epochs):
  total_loss = 0
  for i in range(token_inputs.shape[1]):
    x = token_inputs[i].reshape(-1, 1)
    y = np.array([[token_inputs[0, i]]])

    # Forward and backward pass
    loss = rnn.backward(x, [y], learning_rate)
    total_loss += loss
  
  if epoch % 100 == 0:
      print(f'Epoch {epoch}, Loss: {total_loss}')