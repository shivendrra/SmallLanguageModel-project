"""
Karpathy's BPE code
"""

import os
current_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_directory)
import unicodedata

def get_stats(ids, counts=None):
  counts = {} if counts is None else counts
  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

def merge_vocab(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class BasicTokenizer:
  def __init__(self):
    super().__init__()
  
  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256
  
    text_bytes = text.encode('utf-8')
    ids = list(text_bytes)

    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i in range(num_merges):
      stats = get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = 256 + i
      idx = merge_vocab(ids, pair, idx)
      merges[pair] = idx
      # vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
      
      if verbose:
        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
      
    self.merges = merges
    self.vocab = vocab

  def decode(self, ids):
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text = text_bytes.decode("utf-8", errors="replace")
    return text
  
  def encode(self, text):
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)
    while len(ids) >= 2:
      stats = get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

      if pair not in self.merges:
        break
        
      idx = self.merges[pair]
      ids = merge_vocab(ids, pair, idx)
    
    return ids

# example usage
out_file = '../Data/training_data.txt'
with open(out_file, 'r', encoding='utf-8') as file:
  captions = file.read()

token = BasicTokenizer()
print("training begin--------")
tokenized_data = token.train(captions, vocab_size=512)
print("training ends--------", '\nencoding start-------')
encoded = list(token.encode(captions))
print("encoded data", encoded[:20])
print(list(token.decode(tokenized_data))[:20])