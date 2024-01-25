"""
--> this code takes use of URLFetcher.py and fetches the text data from each of the pages
--> saves it in a .txt file
--> voila!!
"""

# from ...config import DEFAULT_PATH
# import os

# os.chdir(DEFAULT_PATH)

import os
os.chdir('D:/Machine Learning/SLM-Project/')

import json

query_file = 'Data Collection/webscrapper/search_queries.json'
max_limit = 10
with open(query_file, 'r') as file:
  search_queries = json.load(file)

from URLFetcher import generateUrls

fetched_urls = generateUrls(search_queries, max_limit)
out_file = f'Data/webscrapped data/britannica_output.txt'

import requests
from bs4 import BeautifulSoup

def text_extractor(urls, out_file):  
  for url in urls:
    print(url)
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"} 
    r = requests.get(url, headers=headers)
    html_content = r.content
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()

    with open(out_file, 'a', encoding='utf-8') as file:
      file.write(' '.join(text.split(' ')))

text_extractor(fetched_urls, out_file)