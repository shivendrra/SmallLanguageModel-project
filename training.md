### Cooking a model from scratch!!

## Overview

To make a LLM from scratch, we need to make sure a lot of things to be right. It's a very simple yet complex method. If you're one of those people who make Spotify, YouTube, etc app clones and add it to your portfolio, just follow the instructions blindly and build your own LLM, and if you're not like that, you like to understand what's happening behind the curtains, how each and every part works, read this first: [LLMs for noobs]()

If you have already read it, then let's begin...
## Gathering the data

I've made dedicated web-scrapper for the website like britannica & wikipedia along with a transcripts fetcher from youtube. If you use default settings & queries, you can gather around ~5-6Gbs of raw text data, which is enough to train a small model.

### Wikipedia
Wikipedia scrapper has a query file that has around ~40 queries that can fetch upto 2Gbs of data. It's very simple to use it and gather the data from it. This will take some time, probably a lot of time.

```python
from wikipedia import WikiQueries, WikiScraper
queries = WikiQueries()
scrape = WikiScraper()
queries = queries()

output_file = f'../Datasets/wiki_{len(queries)}.txt'
scrape(out_file=output_file, search_queries=queries, extra_urls=True)
```

one more scrapper is there in it, that could gather data from xml dumps from Wikipedia. It can unpack xml & generate target links for the pages and the fetch all the text content and save it in the output file.

```python
from wikipedia import WikiQueries, WikiScraper, WikiXMLScraper

scrape = WikiXMLScraper()
output_file = f'../Datasets/wiki_{len(queries)}.txt'
scrape.scrape(url_file=url_file, out_file=out_file, batch_size=1000)
```

This method is lot more faster and more efficient than the `WikiScrapper()`

### Britannica
Britannica scrapper works the same way as Wikipedia scrapper, it has some default stored queries that could be used to fetch large amount of raw text data. Though it generates less amount of data then Wikipedia, but takes lot less time.

```python
from britannica import Scrapper, searchQueries
sq = searchQueries()
queries = sq()

outfile = f"../Datasets/britannica_{len(queries)}.txt"
bs = Scrapper(search_queries=queries, max_limit=10)
bs(outfile=outfile)
```

### YouTube Transcripts
This was made for generating good transcripts to fine-tune the model, but I figured out how hard it would be to automate the process for a particular kind of fine-tunning dataset, so instead, I made it to fetch raw transcripts.

It's a lengthy process to fetch captions from the videos, and it can go on for long periods. It took me around ~55hrs to fetch 3Gbs of transcripts from over ~167k videos. Data to video ratio isn't any good either, but since I put so much of my efforts in it, just fucking use it!!

```python
from youtube_transcripts import SampleSnippets, TranscriptsCollector
api_key = os.getenv('yt_key')

ss = SampleSnippets()
channe_ids = ss()
target_ids = channe_ids[54:68]
out_file = f'../Datasets/transcripts_{len(target_ids)}.txt'
  
collector = TranscriptsCollector(api_key=api_key)
collector(channel_ids=target_ids, target_file=out_file)
```

Now, the shortcut: Just download the parquet files from [huggingface/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) and convert them into csv & text data using `un-parquet.py` in Data Processing directory. I followed this method. Each time I gathered 4 files, I merged them into a single one using `append_files.py`

## Tokenization

### ***Character Level***
This is a very basic level tokenizer that converts each unique character present in the document into an integer, like a -> 2, b-> 3, 1-> 29, etc. Here's how you can implement it, cause I've not included it in the files.

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

This same tokenizer was used by Karpathy in his tutorial. Easy to understand but not very effective. It has a lots of downsides:
1. **Increased Model Size**: Character-level tokenization results in a larger vocabulary compared to word-level tokenization. This leads to a larger model size and increased memory requirements during training and inference.
2. **Slower Training and Inference**: The larger vocabulary size of character-level tokenization can significantly slow down both training and inference times compared to word-level tokenization. Processing individual characters requires more computations and memory access.
3. **Reduced Semantic Understanding**: Character-level tokenization does not capture word-level semantics or meanings. As a result, the model may struggle to understand and generate coherent text at the word level, leading to potentially less fluent and less coherent output.
4. **Difficulty in Handling Out-of-Vocabulary (OOV) Tokens**: Word-level tokenization typically has mechanisms to handle out-of-vocabulary tokens by using subword tokenization or special tokens like \<unk\>. However, character-level tokenization does not have such mechanisms, making it challenging to handle rare or unseen words during inference.
5. **Loss of Word-Level Information**: By tokenizing at the character level, the model loses access to higher-level linguistic information encoded in words, such as syntactic and semantic relationships between words. This may limit the model's ability to understand and generate text that adheres to grammatical and semantic rules.
6. **Increased Noise in Training Data**: Character-level tokenization may introduce noise into the training data, especially if the corpus contains spelling errors, typos, or non-standard language usage. This can make it more challenging for the model to learn meaningful patterns and relationships in the data.
7. **Less Transferability**: Models trained with character-level tokenization may have limited transferability to downstream tasks compared to models trained with word-level tokenization. Word-level models often capture more generalizable linguistic patterns and can be fine-tuned more effectively for specific tasks.

### ***Sub-word Level***
This kind of tokenizer split's the word into many chunks like:

```
'tokenizer' --> 'token', 'iz', 'er'
```

and fixes all the issue that we faced with the character level tokenization. `Models/tokenizer.py` file contains the code & configurations for the `tiktoken` tokenizer. Use it to tokenize & and train the model.

```python
# Models/tokenizer.py

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
```

## Training Part

I assume you're going to use Google Colab Notebook to train the model, so here is how to do it.
### Prerequisites
1. Gather all the appended files, make sure they are of ~2-2.5Gbs, I'll tell you later why; and upload it on drive. Make sure to get extra space in it.
2. Load the Model Training notebook, or re-write it from scratch while taking help from the original notebook (it would help you to understand the training procedure)
3. Select the GPU, don't go for A100s, T4 & V400s are more than enough, this way you'll save some compute units.
Now since everything is done, we proceed...

Using the notebook, connect it to your Google Drive and download `tiktoken` library. Then load the training dataset and then tokenize it with the set configurations.
After it's done, just run the training cell and then sit back and enjoy, till the model trains. It's that simple.

## Something to remember

1. Keep the `eval_iters` same as the `eval_intervals` and make them both >100 for better training.
2. With a big model, like a 0.5b-1.5b model, if you plan to train it, make sure use learn rate scheduling and keep learn rate = 3e-4 or 3e-5.
3. Don't put `max_iters` more than 5k at a time, this way, you can monitor model's performance and save compute credits.
4. Each time you run new training batch, keep changing `torch.manual_seed()` to ensure that new tokens are fed to the model or change the dataset entirely.
5. Save the model each time when done training, save it in `safetensors` form too.

That's it, have a blast training the model.