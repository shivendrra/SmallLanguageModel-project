"""
--> this code takes use of URLFetcher.py and fetches the text data from each of the pages
--> saves it in a .txt file
--> voila!!
"""

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
headers = {
  'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
  }

import requests
from bs4 import BeautifulSoup
import re

def text_extractor(url_snippet):
  target_url = f"https://britannica.com{url_snippet}"
  r = requests.get(target_url, headers=headers)

  if r.status_code == 200:
    soup = BeautifulSoup(r.content, 'html.parser')
    paragraphs = soup.find_all('p')
    
    # extract text content from each <p> tag, excluding specified text
    page = '\n'.join([p.get_text() for p in paragraphs if "Our editors will review what you’ve submitted and determine whether to revise the article." not in p.get_text()])
    page = re.sub('&\w+;','',page)

    return page

  else:
    print(f"failed to fetch: {target_url}, skipping")

def main(fetched_urls):
  for new_url in fetched_urls:
    r = requests.get(new_url, headers=headers)

    if r.status_code == 200:
      html_content = r.content
      soup = BeautifulSoup(html_content, 'html.parser')
      list_url = soup.find_all('a', attrs={'class': 'font-weight-bold font-18'})
      list_url = [url.get('href') for url in list_url]

      for url_snippet in list_url:
        page = text_extractor(url_snippet)
        with open(out_file, 'a', encoding='utf-8') as file:
          file.write(page)

    else:
      print(f"Failed to fetch the page: {new_url}, skipping")
      continue

if __name__ == "__main__":
  main(fetched_urls)