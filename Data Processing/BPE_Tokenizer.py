import os
os.chdir('D:/Machine Learning/SLM-Project/')

# data for training the BPE
with open('Data/captions.txt', 'r', encoding='utf-8') as file:
  captions = file.read()

# tokenizing
import nltk
token_caps = nltk.word_tokenize(captions)

# train test split
n = int(0.8*len(token_caps))
train_data = token_caps[:n]
val_data = token_caps[n:]

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
num_merges = 10
tokenizer = SubwordTokenizer(num_merges)
tokenizer.learn_bpe(train_data)

# tokenized validation data
tokenized_validation_data = [tokenizer.tokenize(sentence) for sentence in val_data]
# detokenized val data 
detokenized_validation_data = [tokenizer.detokenize(tokens) for tokens in tokenized_validation_data]

# importing data to tokenize
with open('Data/training_data.txt', 'r', encoding='utf-8') as file:
  token_data = file.read()

token_word = nltk.word_tokenize(token_data)
# token_word = token_data

# tokenizing the data
tokenized_corpus = tokenizer.tokenize(token_word)
print(tokenized_corpus[:20])

detokenized_corpus = tokenizer.detokenize(tokenized_corpus)
print(detokenized_corpus[:30])