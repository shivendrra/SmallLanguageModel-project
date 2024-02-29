# from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# # Create a Byte-Pair Encoding (BPE) model
# bpe_model = models.BPE()

# # Initialize the tokenizer
# tokenizer = Tokenizer(bpe_model)

# # Customize pre-tokenization, decoding, and post-processing
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
# tokenizer.decoder = decoders.ByteLevel()

# # Optionally, train the tokenizer on a corpus
# # corpus_files = ["path/to/corpus.txt"]
# # trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# # tokenizer.train(files=corpus_files, trainer=trainer)

# # Tokenize text
# text = "Hello, how are you?"
# encoding = tokenizer.encode(text)
# print("Encoded IDs:", encoding.ids)
# print("Tokens:", encoding.tokens)

# from tokenizers import Tokenizer, models, trainers
# import os
# current_directory = os.path.dirname(os.path.realpath(__file__))
# os.chdir(current_directory)

# # Initialize a BPE model
# bpe_model = models.BPE()

# # Initialize the tokenizer with the BPE model
# tokenizer = Tokenizer(bpe_model)

# out_file = '../../Data/training_data.txt'
# with open(out_file, 'r', encoding='utf-8') as file:
#   training_data = file.read()

# # Train the tokenizer
# tokenizer.train_from_iterator(training_data, trainer=trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]))

# # Tokenize text
# text = """
# It seems that the tokenizer's vocabulary might not have been properly built during training, resulting in only the [UNK] token being present in the vocabulary. This could be due to various reasons, such as incorrect training data or training parameters.
# To troubleshoot this issue, you can try the following steps:
# Verify Training Data: Ensure that the training data provided to the tokenizer is in the correct format and contains a sufficient amount of text for training.
# Adjust Training Parameters: Experiment with different training parameters, such as the size of the vocabulary (vocab_size), the minimum frequency of tokens (min_frequency), and the number of iterations (num_threads) to see if it affects the tokenizer's performance.
# Inspect Training Logs: Check the training logs or outputs to see if there are any error messages or warnings that might indicate issues during training.
# Inspect Tokenizer's Vocabulary: After training, inspect the tokenizer's vocabulary to see if it contains a reasonable number of tokens and if they are representative of the training data.
# By iteratively adjusting these factors and examining the results, you should be able to identify and resolve the issue with the tokenizer's vocabulary generation.
# """
# encoding = tokenizer.encode(text)
# print("Encoded IDs:", encoding.ids)
# print("Tokens:", encoding.tokens)

from tokenizers import Tokenizer, models, trainers
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_directory)
wp_model = models.WordPiece()
tokenizer = Tokenizer(wp_model)

out_file = '../../Data/training_data.txt'
with open(out_file, 'r', encoding='utf-8') as file:
  training_data = file.read()

tokenizer.train_from_iterator(
    [training_data], 
    trainer=trainers.WordPieceTrainer(
        vocab_size=30000,
        min_frequency=2, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
)

text = """
It seems that the tokenizer's vocabulary might not have been properly built during training, resulting in only the [UNK] token being present in the vocabulary. This could be due to various reasons, such as incorrect training data or training parameters.
To troubleshoot this issue, you can try the following steps:
Verify Training Data: Ensure that the training data provided to the tokenizer is in the correct format and contains a sufficient amount of text for training.
Adjust Training Parameters: Experiment with different training parameters, such as the size of the vocabulary (vocab_size), the minimum frequency of tokens (min_frequency), and the number of iterations (num_threads) to see if it affects the tokenizer's performance.
Inspect Training Logs: Check the training logs or outputs to see if there are any error messages or warnings that might indicate issues during training.
Inspect Tokenizer's Vocabulary: After training, inspect the tokenizer's vocabulary to see if it contains a reasonable number of tokens and if they are representative of the training data.
By iteratively adjusting these factors and examining the results, you should be able to identify and resolve the issue with the tokenizer's vocabulary generation.
"""
encoding = tokenizer.encode(text)
print("Encoded IDs:", encoding.ids)
print("Tokens:", encoding.tokens)
