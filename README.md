# Small Language Model

This project is about creating a transformer model, big enough to be called a SLM. Trained on YouTube Videos' Transcript Data and Scrapped data from websites on the internet. It is aimed to learn the concepts behind the transformer model and get a deeper understanding how things work by experimenting on small models.****

## Data Collection:

---

### YouTube Transcripts

YouTube's V3 API is required to fetch results(video ids, urls and captions). Use `Data Collector/transcriptCollector.py` for collecting the data. `Channel_Ids.json` already has more than 45 channels' ids who have available caption data. It will take around 3days to fetch all the transcripts from over 200K videos and file size will be ~3GBs.

### WebScrapping

WebScrapper uses `BeautifulSoup` and `requests` library in python to scrape data from the web, *Britannica.com* in this case. `mainScrapper.py` scrapes data from the website by building custom urls from the `search_queries.json` and then requesting on the url to get the data.

This generates a .txt file of ~600-700MBs approx. You can add more queries and topics for more data.

## Pre-processing & tokenization

---

### Tokenization

- **Character Level**

---

In the initial working of the model, I used character level encoding and tokenization, for the Bi-gram model.

```python
# this is a basic character level tokeinzer

chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoder - decoder
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
```

- **Sub-Word Level**

---

Implemented Byte-Pair Encoding this time using `tit-token` library from **OpenAI.** Wrote a basic training, encoding, decoding and saving model functions and then imported as a module in the code. It works fine.

```python
# Final Models/Transformer/tokenizer.py

class EncoderDecoder:
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
```

## Models

---

### RNN

`Final Models/RNN` directory contains all the necessary codes to run a RNN model, `titoken` library for tokenizing and encoding the data.

### Bi-Gram Model

Made with the help of Karpathy's video '[Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY?si=aHFUrNbYudojGW4j)' with some changes in hyper-parameters and tokenization process, rest is almost same.

### Basic-Transformer

Made it from scratch by looking into various codes and videos. It works, and I trained it till 146million parameters until my GPU crashed. I've to implement some optimizations to run it faster and better than before.

```python
# transformer model

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, eps=norm_eps) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

```

## Contributing

---

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

---

none!
