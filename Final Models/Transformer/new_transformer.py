import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """ generates positional encodings """

    def __init__(self, block_size, n_embd, data):
        super().__init__()
        self.max_seq_len = block_size
        self.n_dim = n_embd
        self.input_seq = data

    def forward(self):
        denominator = torch.pow(10000, self.input_seq / self.n_dim)
        print(denominator.shape)
        positions = torch.arange(self.max_seq_len, dtype=torch.float).reshape(self.max_seq_len, 1)
        print(positions.shape)

        even_pe = torch.sin(positions / denominator)
        odd_pe = torch.cos(positions / denominator)

        # pos_encoding = torch.cat([even_pe, odd_pe], dim=1)
        stacked = torch.stack([even_pe, odd_pe], dim=2)
        pos_encoding = torch.flatten(stacked, start_dim=1, end_dim=2)
        return pos_encoding

class TokenEmbedding:
    def __init__(self, vocab_size, n_embd, data):
      self.vocab_size = vocab_size
      self.n_dim = n_embd
      self.input_data = data

      self.embedding_matrix = nn.Parameter(torch.rand((self.vocab_size, self.n_dim)))
      # self.embedding_matrix = nn.Embedding(self.vocab_size, self.n_embd)

    def forward(self):
      """
        maps input indices to embedding vectors

        Returns:
        - embedded_output: tensor containing the corresponding embedding vectors
      """
      input_indices = self.input_data.long()
      embedded_output = self.embedding_matrix.data[input_indices]

      return embedded_output

input_seq = torch.randn(10)
vocab_size = 10000
block_size = 10
n_dim = 512

token_encodings = TokenEmbedding(vocab_size, n_dim, input_seq).forward() # vocab_size * n_dim = outputs.shape
pos_encodings = PositionalEmbedding(block_size, n_dim, input_seq).forward() # block_size * n_dim = output.shape
print("token:", token_encodings.shape)
print("pos:", pos_encodings.shape)
final_encodings = token_encodings + pos_encodings

print('input sequence:', input_seq)
# print('token encodings:', pos_encodings)
# print('positional embeddings:', pos_encodings)
print('final encodings:', final_encodings)