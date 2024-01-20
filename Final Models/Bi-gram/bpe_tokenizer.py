import os
import nltk
os.chdir('D:/Machine Learning/SLM-Project/')

train_file = 'Data/captions.txt'
with open(train_file, 'r', encoding='utf-8') as infile:
  captions = infile.read()

train_data = """Listen up. Demato might not think you're behind this, but make no mistake, Judge Carmichael, I know a sophisticated
scam when I stop one. - Bravo. - So good. So good. Yeah, riveting. - Christina's performance
there is just, her ferocity. - Fantastic. - Welcome to a sneak peek at this season's most extraordinary and
heart pounding new show, "S.A.F.E Squad" starring
Christina Ricci, and also me, and others. Hi, I'm Stev"""

training_data = ["low", "lower", "newest", "wider"]

from collections import defaultdict

def initialize_vocabulary(data):
  vocab = set(''.join(data))
  return vocab

def get_stats(data):
    # Count the frequency of each pair of consecutive characters
    pair_freq = defaultdict(int)
    for word in data:
        chars = list(word)
        for i in range(len(chars) - 1):
            pair_freq[(chars[i], chars[i+1])] += 1
    return pair_freq

def merge_most_frequent(pair, vocab):
    # Merge the most frequent pair and update the vocabulary
    new_token = ''.join(pair)
    vocab.add(new_token)
    return vocab

def apply_bpe(data, num_merge_iterations):
  vocab = initialize_vocabulary(data)  
  for _ in range(num_merge_iterations):
    pair_freq = get_stats(data)
        
    if not pair_freq:
      break

    most_frequent_pair = max(pair_freq, key=pair_freq.get)
    vocab = merge_most_frequent(most_frequent_pair, vocab)
  return vocab

def tokenize_word(word, vocab):
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

num_merge_iterations = 100
final_vocab = apply_bpe(train_data, num_merge_iterations)
tokenized_words = [tokenize_word(word, final_vocab) for word in captions]

import numpy as np
tokenized_words = np.array(tokenized_words)

print("Final Vocabulary:", final_vocab)
print("Tokenized Words:", tokenized_words[:20])
print(np.shape(tokenized_words))