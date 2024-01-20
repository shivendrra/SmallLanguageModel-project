"""
--> this is a webscarpping code for britannica.com
--> uses a search query list to generate a target url
--> fetches the actual url of pages contatning data about the search query topics
"""
import os
os.chdir('D:/Machine Learning/SLM-Project/')

import requests
from bs4 import BeautifulSoup

def scrapeUrls(query, pgNo):
  formattedQuery = '%20'.join(query.split(' '))
  url = f"https://www.britannica.com/search?query={formattedQuery}&page={pgNo}"
  return url

def generateUrls(search_queries, max_limit):
  links = []
  for query in search_queries:
    pageNo = 1
    for i in range(max_limit):
      links.append(scrapeUrls(query, pageNo))
      pageNo += 1
  return links

def getUrls(url):
  headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"} 
  r = requests.get(url, headers=headers)
  links = []
  soup = BeautifulSoup(r.content, 'html.parser')

  for li in soup.select('.search-results.col ul.list-unstyled.results li'):
    link_element = li.select_one('a.font-weight-bold')
    href = link_element.get('href')
    links.append(href)
    print(links)
  return links

if __name__ == "__main__":
  urls = generateUrls()
  for url in urls:
    getUrls(url)