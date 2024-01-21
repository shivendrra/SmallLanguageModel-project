from tokenizer import SubwordTokenizer

class EncoderDecoder:
  def __init__(self, input_data, n_iters):
    self.n_iters = n_iters
    self.input_data = input_data
    self.subt = SubwordTokenizer(self.n_iters)
    self.subt.apply_bpe(input_data)
    self.vocab = self.subt.vocab 

  def encoder(self):
    token_data = self.subt.tokenize_data(self.input_data, self.vocab)
    stoi = { ch:i for i,ch in enumerate(self.vocab) }
    encode = lambda s: [stoi[c] for c in s]

    encoded_text = encode(token_data)
    return encoded_text

  def decoder(self, tokens):
    itos = { i:ch for i,ch in enumerate(self.vocab) }
    decode = lambda l: ''.join([itos[i] for i in l])
    decoded_text = decode(tokens)

    detoken_text = self.subt.detokenize_data(decoded_text)
    return detoken_text