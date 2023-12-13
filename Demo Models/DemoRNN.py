import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

os.chdir('D:/Machine Learning/SLM-Project')
with open('Data/vector_data.csv', 'r') as file:
    data = file.read()
data = pd.read_csv(data)

# Extract values as a NumPy array
tfidf_matrix = data.values.T

# Assuming tfidf_matrix is your TF-IDF matrix
reference_document_index = 0
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
    
    def generate(self, seed, num_tokens):
        # Generate tokens using the trained RNN
        generated_sequence = []

        # Initialize hidden state
        h = np.zeros((self.hidden_size, 1))

        # Use the seed as the first input
        x = seed.reshape(-1, 1)

        for _ in range(num_tokens):
            # Forward pass through the RNN
            h = sigmoid(np.dot(self.W_ih, x) + np.dot(self.W_hh, h) + self.b_h)
            output = sigmoid(np.dot(self.W_ho, h) + self.b_o)

            # Append the generated token to the sequence
            generated_sequence.append(output.flatten())

            # Use the generated output as the input for the next time step
            x = output

        return generated_sequence

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
    # print(f'Input: {x.flatten()}, Output: {output.flatten()}')

seed_token = tfidf_matrix[:, 0].reshape(-1, 1)
num_generated_tokens = 4

from sklearn.feature_extraction.text import TfidfVectorizer as vect

# Generate tokens using the trained RNN
generated_tokens = rnn.generate(seed_token, num_generated_tokens)

# Convert the generated tokens back to text using the original TF-IDF vectorizer
inverse_tfidf_matrix = vect.inverse_transform(generated_tokens)

# Convert the TF vectors back to text
reconstructed_text_data = [' '.join(words) for words in inverse_tfidf_matrix]

# Print the reconstructed text data
print("Reconstructed Text Data:")
print(reconstructed_text_data)
