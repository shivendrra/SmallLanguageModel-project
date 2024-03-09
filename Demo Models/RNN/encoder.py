from tokenizer import SubwordTokenizer

class EncoderDecoder:
  def __init__(self, train_data, n_iters):
    self.n_iters = n_iters
    self.train_data = train_data
    self.subt = SubwordTokenizer(self.n_iters)
    self.subt.apply_bpe(self.train_data)
    self.vocab = self.subt.vocab

  def encoder(self, input_data):
    encoded_text = []
    words = input_data.split()
    padding_token = '</w>'
    self.vocab[padding_token] = len(self.vocab)
    stoi = {ch: i for i, ch in enumerate(self.vocab)}
        
    for i, word in enumerate(words):
      # Add padding token between words (except for the first word)
      if i > 0:
        encoded_text.append(stoi[padding_token])

      token_data = self.subt.tokenize_data(word, self.vocab)
      encoded_text.extend([stoi[c] for c in token_data])
    return encoded_text

  def decoder(self, tokens):
    itos = {i: ch for i, ch in enumerate(self.vocab)}
    decode = lambda l: ''.join([itos.get(int(i.item()), ' ').replace('</w>', ' ') for i in l])
    decoded_text = decode(tokens)

    detoken_text = self.subt.detokenize_data(decoded_text)
    return detoken_text