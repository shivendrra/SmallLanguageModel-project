import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w_ih = np.random.randn(hidden_size, input_size)
        self.w_hh = np.random.randn(hidden_size, hidden_size)
        self.w_oh = np.random.randn(output_size, hidden_size)

        self.b_h = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((output_size, 1))

        self.h = np.zeros((hidden_size, 1))

    def forward(self, x_seq):
        outputs = []
        for x in x_seq:
            x = x.reshape(-1, 1)
            self.h = sigmoid(np.dot(self.w_ih, x) + np.dot(self.w_hh, self.h) + self.b_h)
            output = sigmoid(np.dot(self.w_oh, self.h) + self.b_o)
            outputs.append(output)
        return outputs

    def backward(self, x_seq, y_seq, learning_rate):
        total_err = 0
        for t in range(len(x_seq)):
            x = x_seq[t]
            y = y_seq[0][t]  

            # forward pass
            output = self.forward(x_seq[:t + 1])[-1]

            # backward pass
            error = y - output
            total_err += np.mean((error) ** 2)

            d_output = error * sigmoid_derivative(output)
            d_w_oh = np.dot(d_output, self.h.T)
            d_b_o = d_output

            # hidden layer gradients
            d_h = np.dot(self.w_oh.T, d_output) * sigmoid_derivative(self.h)
            d_w_ih = np.dot(d_h, x.T)
            d_w_hh = np.dot(d_h, self.h.T)
            d_b_h = d_h

            self.w_ih += learning_rate * d_w_ih
            self.w_hh += learning_rate * d_w_hh
            self.w_oh += learning_rate * d_w_oh
            self.b_h += learning_rate * d_b_h
            self.b_o += learning_rate * d_b_o

        return total_err