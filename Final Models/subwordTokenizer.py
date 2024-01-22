from tokenizers import normalizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, StripAccents

import os

current_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_directory)

class CustomEncoderDecoder:
  def __init__(self, model_path="custom-model.json"):
    self.tokenizer = Tokenizer(models.BPE())
    self.model_path = os.path.join(current_directory, model_path)
    self.setup_tokenizer()

  def train_tokenizer(self, corpus, vocab_size=1000):
    trainer = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>", "<s>", "<\s>"], vocab_size=vocab_size)
    self.tokenizer.train_from_iterator(corpus, trainer=trainer)
    self.save_model()

  def save_model(self):
    model_directory = os.path.dirname(self.model_path)
    os.makedirs(model_directory, exist_ok=True)
    print("Model Path:", self.model_path)
    self.tokenizer.model.save(model_directory, "custom-model")

  def load_model(self):
    self.tokenizer.model = models.BPE.load(self.model_path)

  def setup_tokenizer(self):
    self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    self.tokenizer.decoder = decoders.ByteLevel()
    self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    self.tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    self.tokenizer.enable_truncation(max_length=512)

  def encode(self, text):
    encoding = self.tokenizer.encode(text)
    return encoding.ids

  def decode(self, ids):
    tokens = self.tokenizer.decode(ids)
    return tokens

# Example usage:
corpus = ["This is an example sentence.", "Another example sentence."]

# Initialize encoder-decoder
encoder_decoder = CustomEncoderDecoder()

# Train tokenizer
encoder_decoder.train_tokenizer(corpus, vocab_size=1000)

# Encode and decode
encoded_ids = encoder_decoder.encode("This is an example sentence.")
decoded_text = encoder_decoder.decode(encoded_ids)

print(f"Encoded IDs: {encoded_ids}")
print(f"Decoded Text: {decoded_text}")
