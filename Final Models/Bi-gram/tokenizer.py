import tiktoken

pre_encodings = 'p50k_base'
pre_model = 'text-davinci-003'
class Tokenizer:
  def __init__(self, encoding=None, model=None):
    self.encodings = encoding if encoding is not None else pre_encodings
    self.model = model if model is not None else pre_model
    self.tokenizer = tiktoken.get_encoding(self.encodings)
    self.tokenizer = tiktoken.encoding_for_model(self.model)
  
  def encode(self, data):
    return self.tokenizer.encode(data)
  
  def decode(self, tokens):
    return self.tokenizer.decode(tokens)
  
  def get_vocab(self):
    return self.tokenizer.n_vocab