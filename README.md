# Small Language Model
This project is about creating a transformer model, big enough to be called a SLM. Trained on YouTube Videos' Transcript Data and Scrapped data from websites on the internet.

## Data Collection:

#### YouTube Transcripts
YouTube's V3 API is required to fetch results(video ids, urls and captions). Use `Data Collector/transcriptCollector.py` for collecting the data. `Channel_Ids.json` already has more than 45 channels' ids who have available caption data. It will take around 3days to fetch all the transcripts from over 200K videos and file size will be ~3GBs.

#### WebScrapping
WebScrapper uses `BeautifulSoup` and `requests` library in python to scrape data from the web, _Britannica.com_ in this case. `mainScrapper.py` scrapes data from the website by building custom urls from the `search_queries.json` and then requesting on the url to get the data.

This generates a .txt file of ~600-700MBs approx. You can add more queries and topics for more data.

## Models

#### RNN
`Final Models/RNN` directory contains all the necessary codes to run a RNN model, `titoken` library for tokenizing and encoding the data.

#### Bi-Gram Model
Made with the help of Karpathy's video '[Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY?si=aHFUrNbYudojGW4j)' with some changes in hyper-parameters and tokenization process, rest is almost same.

#### Basic-Transformer
Made it from scratch by looking into various codes and videos. It works, and I trained it till 146million parameters until my GPU crashed. I've to implement some optimizations to run it faster and better than before.


## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
none!
