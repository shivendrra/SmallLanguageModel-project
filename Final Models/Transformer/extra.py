import os
os.chdir('D:/Machine Learning/SLM-Project')
import timeit
start_load = timeit.default_timer()

print('code running')
with open('Data/txt files/big_data_v2.txt', 'r', encoding='utf-8') as file:
  captions = file.read()
  print('file loaded')

start = timeit.default_timer()

from tokenizer import EncoderDecoder
ed = EncoderDecoder()
# ed.train_tokenizer(captions, vocab_size=40000)
ed.load_model()
vocab_size = len(ed.tokenizer.get_vocab())
input_data = ed.encode(captions)

end = timeit.default_timer()

print(f"total words in dataset {len(captions) / 1e9} billion")
print(f"total no of tokens {len(input_data) / 1e6} million")
print(f"vocab size is: {vocab_size}")
print(f"time to load file {(end - start_load) / 60} mins")
print(f"total time taken to tokenize {(end - start) / 60} mins")