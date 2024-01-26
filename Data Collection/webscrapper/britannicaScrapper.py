"""
--> this code takes use of URLFetcher.py and fetches the text data from each of the pages
--> saves it in a .txt file
--> voila!!
"""

import os
import json
os.chdir('D:/Machine Learning/SLM-Project/')


query_file = 'Data Collection/webscrapper/search_queries.json'
out_file = f'Data/webscrapped data/britannica_output.txt'
max_limit = 10

with open(query_file, 'r') as file:
  search_queries = json.load(file)

print('fetching snippets from queries')

from URLFetcher import BritannicaUrls
scrape = BritannicaUrls(search_queries=search_queries, max_limit=10)
url_sinppets = scrape.generate_urls()

print('fetched snippets successfully!')
print('scrapping and saving the data-------')

import requests
from bs4 import BeautifulSoup
import re

def text_extractor(url_snippet):
  target_url = f"https://britannica.com{url_snippet}"
  r = requests.get(target_url, headers=scrape.headers)

  if r.status_code == 200:
    soup = BeautifulSoup(r.content, 'html.parser')
    paragraphs = soup.find_all('p')
    
    # extract text content from each <p> tag, excluding specified text
    page = '\n'.join([p.get_text() for p in paragraphs if "Our editors will review what you’ve submitted and determine whether to revise the article." not in p.get_text()])
    page = re.sub('&\w+;','',page)

    return page

  else:
    print(f"failed to fetch: {target_url}, skipping")

if __name__ == '__main__':
  for snippets in url_sinppets:
    page = text_extractor(snippets)
    with open(out_file, 'a', encoding='utf-8') as file:
      file.write(page)