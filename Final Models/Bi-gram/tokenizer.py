"""
Byte Pair Encoding sub-word tokenizer, without encoder or decoder
"""

from collections import defaultdict

class SubwordTokenizer:
  def __init__(self, n_merges):
    self.n_merges = n_merges
    self.vocab = None

  def get_stats(self, vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
      symbols = word.split()
      for i in range(len(symbols) - 1):
        pairs[symbols[i], symbols[i+1]] += freq

    return pairs
    
  def merge_vocab(self, pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
      new_word = word.replace(bigram, replacement)
      new_vocab[new_word] = vocab[word]

    return new_vocab
    
  def apply_bpe(self, data):
    vocab = defaultdict(int)

    for word in data:
      vocab[' '.join(list(word))] += 1
    
    for i in range(self.n_merges):
      pairs = self.get_stats(vocab)
      if not pairs:
        break
    
      best_pair = max(pairs, key=pairs.get)
      vocab = self.merge_vocab(best_pair, vocab)
    
    self.vocab = vocab
    return vocab

  def tokenize_data(self, word, vocab):
    tokens = []
    while word:
      found = False
      for token in vocab:
        if word.startswith(token):
          tokens.append(token)
          word = word[len(token):]
          found = True
          break
      if not found:
        tokens.append(word[0])
        word = word[1:]
    return tokens
    
  def detokenize_data(self, tokens):
    detokenized_txt = ''.join(tokens)
    return detokenized_txt
    
  def validate(self, val_data):
    correct_tokens = 0
    total_samples = 0

    for text in val_data:
      token_txt = self.tokenize_data(text, self.vocab)
      detoken_txt = self.detokenize_data(token_txt)      
    
      if detoken_txt == text:
        correct_tokens += 1
      total_samples += 1
    
    accuracy = correct_tokens / total_samples
    return accuracy