import os
os.chdir('D:/Machine Learning/SLM-Project')
import timeit
start_load = timeit.default_timer()

print('code running')
with open('Data/txt files/big_data_v2.txt', 'r', encoding='utf-8') as file:
  captions = file.read()
  print('file loaded')

start = timeit.default_timer()

from tokenizer import EncoderDecoder
ed = EncoderDecoder()
# ed.train_tokenizer(captions, vocab_size=40000)
ed.load_model()
vocab_size = len(ed.tokenizer.get_vocab())
input_data = ed.encode(captions)

end = timeit.default_timer()

print(f"total words in dataset {len(captions) / 1e9} billion")
print(f"total no of tokens {len(input_data) / 1e6} million")
print(f"vocab size is: {vocab_size}")
print(f"time to load file {(end - start_load) / 60} mins")
print(f"total time taken to tokenize {(end - start) / 60} mins")

# import torch
# import torch.nn as nn
# import numpy as np

# class PositionalEmbedding(nn.Module):
#     """ generates positional encodings """

#     def __init__(self, block_size, n_embd, data):
#         super().__init__()
#         self.max_seq_len = block_size
#         self.n_dim = n_embd
#         self.input_seq = data
#         print(f"max_seq: {self.max_seq_len} \nn_dim: {self.n_dim} \ninput_seq: {np.shape(self.input_seq)}")

#     def forward(self):
#         denominator = torch.pow(10000, self.input_seq / self.n_dim)
#         print("denominator", denominator.shape)
#         positions = torch.arange(self.max_seq_len, dtype=torch.float).reshape(self.max_seq_len, 1)
#         print("positions", positions.shape)
#         even_pe = torch.sin(positions / denominator)
#         print("even pe:",even_pe.shape)
#         odd_pe = torch.cos(positions / denominator)
#         print("odd pe:",odd_pe.shape)

#         stacked = torch.stack([even_pe, odd_pe], dim=2)
#         print("stacked:", stacked.shape)
#         pos_encoding = torch.flatten(stacked, start_dim=1, end_dim=2)
#         print("pos encoding:", pos_encoding.shape)
#         return pos_encoding

# class TokenEncodings:
#     def __init__(self, vocab_size, n_embd, data):
#         self.vocab_size = vocab_size
#         self.n_dim = n_embd
#         self.input_data = data

#     def forward(self):
#         embeddings = torch.randn(self.vocab_size, self.n_dim)
#         embeddings /= torch.norm(embeddings, dim=1, keepdim=True)
#         return embeddings

# # Example usage
# input_seq = torch.randn(700)
# vocab_size = 100
# block_size = 100
# n_dim = 512

# token_encodings = TokenEncodings(vocab_size, n_dim, input_seq).forward() # vocab_size * n_dim = outputs.shape
# pos_embeddings = PositionalEmbedding(block_size, n_dim, input_seq).forward() # block_size * n_dim = output.shape
# print("\n----------------\n")
# print("token:", np.shape(token_encodings))
# print("pos:", np.shape(pos_embeddings))
# final_encodings = token_encodings + pos_embeddings

# print('input sequence:', input_seq)
# print('token encodings:', token_encodings)
# print('positional embeddings:', pos_embeddings)
# print('final encodings:', final_encodings)

# import torch
# import torch.nn as nn

# class PositionalEmbedding(nn.Module):
#     """Generates positional encodings"""

#     def __init__(self, block_size, n_embd):
#         super().__init__()
#         self.max_seq_len = block_size
#         self.n_dim = n_embd

#     def forward(self, input_seq):
#         denominator = torch.pow(10000, torch.arange(self.n_dim, dtype=torch.float) / self.n_dim)
#         positions = torch.arange(self.max_seq_len, dtype=torch.float).reshape(self.max_seq_len, 1)
#         even_pe = torch.sin(positions / denominator)
#         odd_pe = torch.cos(positions / denominator)

#         # stacked = torch.stack([even_pe, odd_pe], dim=2)
#         # pos_encoding = torch.flatten(stacked, start_dim=1, end_dim=2)
#         # pos_encoding = torch.cat([even_pe, odd_pe], dim=1)[:, :self.n_dim]

#         pos_encoding = torch.cat([even_pe, odd_pe], dim=1)[:, :self.n_dim]

#         return pos_encoding

# class TokenEncodings:
#     def __init__(self, vocab_size, n_embd):
#         self.vocab_size = vocab_size
#         self.n_dim = n_embd

#     def forward(self, input_seq):
#         embeddings = torch.randn(self.vocab_size, self.n_dim)
#         embeddings /= torch.norm(embeddings, dim=1, keepdim=True)
#         return embeddings

# # Example usage
# vocab_size = 512
# block_size = 64
# n_dim = 32

# token_encodings = TokenEncodings(vocab_size, n_dim).forward(None)  # No input_seq needed
# pos_embeddings = PositionalEmbedding(block_size, n_dim).forward(None)  # No input_seq needed
# print("token:", token_encodings.shape)
# print("pos:", pos_embeddings.shape)
# final_encodings = token_encodings + pos_embeddings

# print('token encodings:', token_encodings)
# print('positional embeddings:', pos_embeddings)
# print('final encodings:', final_encodings)

# import torch

# torch.manual_seed(1400)
# class CustomModel:
#     def __init__(self, vocab_size, block_size, n_embd):
#         self.vocab_size = vocab_size
#         self.block_size = block_size
#         self.n_embd = n_embd

#         # Initialize token embedding table
#         self.token_embedding_table = self.initialize_token_embeddings()

#         # Initialize positional embedding table
#         self.position_embedding_table = self.initialize_position_embeddings()

#     def initialize_token_embeddings(self):
#         # Initialize token embeddings randomly
#         token_embeddings = torch.randn(self.vocab_size, self.n_embd)
#         # Normalize the embeddings along the embedding dimension
#         token_embeddings /= torch.norm(token_embeddings, dim=1, keepdim=True)
#         return token_embeddings

#     def initialize_position_embeddings(self):
#         # Initialize positional embeddings using sine and cosine functions
#         position_embeddings = torch.zeros(self.block_size, self.n_embd)
#         for pos in range(self.block_size):
#             for i in range(0, self.n_embd, 2):
#                 position = torch.tensor(pos, dtype=torch.float)
#                 exponent = torch.tensor(i / self.n_embd, dtype=torch.float)
#                 position_embeddings[pos, i] = torch.sin(position / (10000 ** exponent))
#                 position_embeddings[pos, i + 1] = torch.cos(position / (10000 ** exponent))
#         return position_embeddings

# # Example usage
# vocab_size = 1000
# block_size = 512
# n_embd = 256

# model = CustomModel(vocab_size, block_size, n_embd)

# # Access token embedding table
# token_embeddings = model.token_embedding_table
# print("Token Embeddings Shape:", token_embeddings.shape)

# # Access positional embedding table
# position_embeddings = model.position_embedding_table
# print("Positional Embeddings Shape:", position_embeddings.shape)