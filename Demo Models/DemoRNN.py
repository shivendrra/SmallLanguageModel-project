import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

os.chdir('D:/Machine Learning/SLM-Project')
data = pd.read_csv("Data/vector_data.csv")

# Extract values as a NumPy array
tfidf_matrix = data.values.T

# Assuming tfidf_matrix is your TF-IDF matrix
reference_document_index = 0  # Choose a reference document
reference_vector = tfidf_matrix[:, reference_document_index].reshape(1, -1)

# Compute cosine similarity scores
similarity_scores = cosine_similarity(tfidf_matrix.T, reference_vector)

# Normalize similarity scores to [0, 1]
normalized_scores = (similarity_scores - np.min(similarity_scores)) / (np.max(similarity_scores) - np.min(similarity_scores))

# Set target values
your_target_values_for_each_document = normalized_scores.flatten()

# Print normalized scores
print("Normalized Similarity Scores:")
print(your_target_values_for_each_document)

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the RNN class
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_ih = np.random.randn(hidden_size, input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size)
        self.W_ho = np.random.randn(output_size, hidden_size)

        self.b_h = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((output_size, 1))

        # Initialize hidden state
        self.h = np.zeros((hidden_size, 1))

    def forward(self, x):
        # Forward pass through the RNN
        self.h = sigmoid(np.dot(self.W_ih, x) + np.dot(self.W_hh, self.h) + self.b_h)
        output = sigmoid(np.dot(self.W_ho, self.h) + self.b_o)
        return output

    def backward(self, x, y, output, learning_rate):
        # Backward pass through the RNN
        error = y - output

        # Output layer gradients
        d_output = error * sigmoid_derivative(output)
        d_W_ho = np.dot(d_output, self.h.T)
        d_b_o = d_output

        # Hidden layer gradients
        d_h = np.dot(self.W_ho.T, d_output) * sigmoid_derivative(self.h)
        d_W_ih = np.dot(d_h, x.T)
        d_W_hh = np.dot(d_h, self.h.T)
        d_b_h = d_h

        # Update weights and biases
        self.W_ih += learning_rate * d_W_ih
        self.W_hh += learning_rate * d_W_hh
        self.W_ho += learning_rate * d_W_ho
        self.b_h += learning_rate * d_b_h
        self.b_o += learning_rate * d_b_o

# Load TF-IDF vectors
tfidf_matrix = np.loadtxt('vector_data.csv', delimiter=',', skiprows=1)

# Transpose the TF-IDF matrix to have a shape of (number_of_features, number_of_documents)
tfidf_matrix = tfidf_matrix.T

# Reshape TF-IDF vectors to have a shape of (number_of_features, 1)
tfidf_matrix = tfidf_matrix.reshape(tfidf_matrix.shape[0], 1)

# Example usage
input_size = tfidf_matrix.shape[0]
hidden_size = 4
output_size = 1

# Create RNN
rnn = SimpleRNN(input_size, hidden_size, output_size)

# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    total_loss = 0
    for i in range(tfidf_matrix.shape[1]):
        x = tfidf_matrix[:, i].reshape(-1, 1)
        y = np.array([[your_target_values_for_each_document[i]]])  # Replace with your target values

        # Forward pass
        output = rnn.forward(x)

        # Backward pass
        rnn.backward(x, y, output, learning_rate)

        # Compute loss (mean squared error)
        total_loss += np.mean((output - y) ** 2)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')

# Test the trained model
for i in range(tfidf_matrix.shape[1]):
    x = tfidf_matrix[:, i].reshape(-1, 1)
    output = rnn.forward(x)
    print(f'Input: {x.flatten()}, Output: {output.flatten()}')


# # Define sigmoid activation function
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Define derivative of sigmoid function
# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Define the RNN class
# class SimpleRNN:
#     def __init__(self, input_size, hidden_size, output_size):
#         # Initialize weights and biases
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         self.W_ih = np.random.randn(hidden_size, input_size)
#         self.W_hh = np.random.randn(hidden_size, hidden_size)
#         self.W_ho = np.random.randn(output_size, hidden_size)

#         self.b_h = np.zeros((hidden_size, 1))
#         self.b_o = np.zeros((output_size, 1))

#         # Initialize hidden state
#         self.h = np.zeros((hidden_size, 1))

#     def forward(self, x_sequence):
#         # Forward pass through the RNN
#         outputs = []
#         for x in x_sequence:
#             self.h = sigmoid(np.dot(self.W_ih, x) + np.dot(self.W_hh, self.h) + self.b_h)
#             output = sigmoid(np.dot(self.W_ho, self.h) + self.b_o)
#             outputs.append(output)
#         return outputs

#     def backward(self, x_sequence, y_sequence, learning_rate):
#         # Backward pass through the RNN
#         total_error = 0
#         for t in range(len(x_sequence)):
#             x = x_sequence[t]
#             y = y_sequence[t]

#             # Forward pass
#             output = self.forward([x])[0]

#             # Backward pass
#             error = y - output
#             total_error += np.mean((error) ** 2)

#             # Output layer gradients
#             d_output = error * sigmoid_derivative(output)
#             d_W_ho = np.dot(d_output, self.h.T)
#             d_b_o = d_output

#             # Hidden layer gradients
#             d_h = np.dot(self.W_ho.T, d_output) * sigmoid_derivative(self.h)
#             d_W_ih = np.dot(d_h, x.T)
#             d_W_hh = np.dot(d_h, self.h.T)
#             d_b_h = d_h

#             # Update weights and biases
#             self.W_ih += learning_rate * d_W_ih
#             self.W_hh += learning_rate * d_W_hh
#             self.W_ho += learning_rate * d_W_ho
#             self.b_h += learning_rate * d_b_h
#             self.b_o += learning_rate * d_b_o

#         return total_error

# # Load TF-IDF vectors
# tfidf_matrix = np.loadtxt('vector_data.csv', delimiter=',', skiprows=1)

# # Transpose the TF-IDF matrix to have a shape of (number_of_features, number_of_documents)
# tfidf_matrix = tfidf_matrix.T

# # Reshape TF-IDF vectors to have a shape of (number_of_features, 1)
# tfidf_matrix = tfidf_matrix.reshape(tfidf_matrix.shape[0], 1)

# # Example usage
# input_size = tfidf_matrix.shape[0]
# hidden_size = 4
# output_size = 1

# # Create RNN
# rnn = SimpleRNN(input_size, hidden_size, output_size)

# # Training loop
# epochs = 10000
# learning_rate = 0.1

# for epoch in range(epochs):
#     total_loss = 0
#     for i in range(tfidf_matrix.shape[1]):
#         x = tfidf_matrix[:, i].reshape(-1, 1)
#         y = np.array([[your_target_values_for_each_document[i]]])

#         # Forward and backward pass
#         loss = rnn.backward(x, [y], learning_rate)
#         total_loss += loss

#     if epoch % 1000 == 0:
#         print(f'Epoch {epoch}, Loss: {total_loss}')

# # Test the trained model
# for i in range(tfidf_matrix.shape[1]):
#     x = tfidf_matrix[:, i].reshape(-1, 1)
#     outputs = rnn.forward([x])
#     print(f'Input: {x.flatten()}, Output: {outputs[0][0].flatten()}')
