import os
import nltk
os.chdir('D:/Machine Learning/SLM-Project/')

# data for training the BPE
with open('Data/training_data.txt', 'r', encoding='utf-8') as file:
  captions = file.read()
token_caps = nltk.word_tokenize(captions)

# train test split for training data
n = int(0.8*len(token_caps))
bpe_train_data = token_caps[:n]
bpe_val_data = token_caps[n:]

# creating a sub-word level tokenizer
import re
from collections import defaultdict

class SubwordTokenizer:
  def __init__(self, num_merges):
    self.num_merges = num_merges
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

  def learn_bpe(self, data):
    vocab = defaultdict(int)
    for word in data:
      vocab[' '.join(list(word)) + ' </w>'] += 1

    for i in range(self.num_merges):
      pairs = self.get_stats(vocab)
      if not pairs:
        break
      best_pair = max(pairs, key=pairs.get)
      vocab = self.merge_vocab(best_pair, vocab)

    self.vocab = vocab

  def tokenize(self, text):
    tokens = []
    if isinstance(text, list):  # check if text is a list
      for word in text:
        word = ' '.join(list(word)) + ' </w>'
        while word:
          if word in self.vocab:
            tokens.append(word)
            break
          else:
            tokens.append(word[:2])
            word = word[2:]
    else:  # where text is a string
      word = ' '.join(list(text)) + ' </w>'
      while word:
        if word in self.vocab:
          tokens.append(word)
          break
        else:
          tokens.append(word[:2])
          word = word[2:]
    return tokens
  
  def detokenize(self, tokens):
    detokenized_txt = ''.join(tokens)
    detokenized_txt = detokenized_txt.replace('</w>', '')
    return detokenized_txt
  
  def validate(self, val_data):
    correct_tokens = 0
    total_samples = 0

    for text in val_data:

      token_txt = self.tokenize(text)
      detoken_txt = self.detokenize(token_txt)

      if detoken_txt == text:
        correct_tokens += 1

      total_samples +=1

    accuracy = correct_tokens / total_samples
    print(f"accuracy is {accuracy*100} %")
    return accuracy

# Train the tokenizer on the training data
num_merges = 30
tokenizer = SubwordTokenizer(num_merges)
tokenizer.learn_bpe(bpe_train_data)

# tokenized validation data
tokenized_validation_data = [tokenizer.tokenize(sentence) for sentence in bpe_val_data]
# detokenized val data 
detokenized_validation_data = [tokenizer.detokenize(tokens) for tokens in tokenized_validation_data]

# data for tokenization
with open('Data/new_training_data.txt', 'r', encoding='utf-8') as file:
  data = file.read()

# train-test split for token data
i = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:i]
val_data = data[i:]

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# tokenizing the data seperately
# normal tokenization first
train_word = nltk.word_tokenize(train_data)
val_word = nltk.word_tokenize(val_data)

# sub-word tokenization
train_data = tokenizer.tokenize(train_word)
val_data = tokenizer.tokenize(val_word)

with open('Data/tokenized_train_data.txt', 'w', encoding='utf-8') as file:
  file.write(str(train_data))

with open('Data/tokenized_val_data.txt', 'w', encoding='utf-8') as file:
  file.write(str(val_data))

print('file written successfully')
# tokenizing the data
print(train_data[:10])
print(val_data[:10])