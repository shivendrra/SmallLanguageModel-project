import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  """ generates positional encodings """

  def __init__(self, block_size, n_embd) -> None:
    super().__init__()
    self.max_seq_len = block_size
    self.n_dim = n_embd

  def forward(self):
    even_i = torch.arange(0, self.n_dim, 2).float()
    denominator = torch.pow(10000, even_i / self.n_dim)
    positions = torch.arange(self.max_seq_len, dtype=torch.long).reshape(self.max_seq_len, 1)
    
    even_pe = torch.sin(positions / denominator)
    odd_pe = torch.cos(positions / denominator)

    stacked = torch.stack([even_pe, odd_pe], dim=2)
    pos_encoding = torch.flatten(stacked, start_dim=1, end_dim=2)
    return pos_encoding

pe = PositionalEncoding(10, 5)
outputs = pe.forward()
print(outputs)