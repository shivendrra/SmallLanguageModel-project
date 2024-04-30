# SmallLanguageModel

This repository contains all the necessary items needed to build your own LLM from scratch. Just follow the instructions. Inspired from Karpathy's nanoGPT and Shakespeare generator, I made this repository to build my own LLM. It has everything from data collection for the Model to architecture file, tokenizer and train file.

## Repo Structure
This repo contains:
1. **Data Collector:** Web-Scrapper containing directory, in case you want to gather the data from scratch instead of downloading.
2. **Data Processing:** Directory that contains code to pre-process certain kinds of file like converting parquet files to .txt and .csv files and file appending codes.
3. **Models:** Contains all the necessary code to train a model of your own. A BERT model, GPT model & Seq-2-Seq model along with tokenizer and run files.

## Prerequisites
Before setting up SmallLanguageModel, ensure that you have the following prerequisites installed:

1. Python 3.8 or higher
2. pip (Python package installer)

## How to use:
Follow these steps to train your own tokenizer or generate outputs from the trained model:
1. Clone this repository:
	```shell
	git clone https://github.com/shivendrra/SmallLanguageModel-project
	cd SLM-clone
	```

2. Install Dependencies:
	```shell
	pip install requirements.txt
	```

3. Train:
	Read the [training.md](https://github.com/shivendrra/SmallLanguageModel-project/blob/main/training.md) for more information. Follow it.

## StarHistory

[![Star History Chart](https://api.star-history.com/svg?repos=shivendrra/SmallLanguageModel-project&type=Date)](https://star-history.com/#shivendrra/SmallLanguageModel-project&Date)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
MIT License. Check out [License.md](https://github.com/shivendrra/SmallLanguageModel-project/blob/main/LICENSE) for more info.