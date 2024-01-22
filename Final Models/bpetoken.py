from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    replacement_freq = vocab.get(pair[0], 0) + vocab.get(pair[1], 0)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    new_vocab[replacement] = replacement_freq
    return new_vocab

def learn_bpe(text, num_merges):
    vocab = defaultdict(int)
    for word in text.split():
        vocab[' '.join(list(word)) + ' </w>'] += 1

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)

    return vocab

def tokenize(text, vocab):
    text = text + ' </w>'
    symbols = text.split()
    for i in range(len(symbols)):
        if symbols[i] in vocab:
            symbols[i] = symbols[i]
        else:
            symbols[i] = '<unk>'
    return ' '.join(symbols)

def detokenize(tokens, pad_token='</w>', unk_token='<unk>'):
    detokenized_text = " ".join(token.replace(pad_token, '</w>').replace(unk_token, '') for token in tokens)
    return detokenized_text.strip()

# Example usage:
import os
os.chdir("d:/Machine Learning/SLM-Project/")
with open('Data/training_data.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()
num_merges = 10

with open('Data/captions.txt', 'r', encoding='utf-8') as file:
  data = file.read()


bpe_vocab = learn_bpe(corpus, num_merges)
tokenized_text = tokenize(data, bpe_vocab)

# print("BPE Vocabulary:")
# print(bpe_vocab, "\n", len(bpe_vocab))
print("\nTokenized Text:")
print(tokenized_text)

detokenized_text = detokenize(tokenized_text)
print("\nDetokenized Text:")
print(detokenized_text)