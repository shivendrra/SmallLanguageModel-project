"""
  --> generates a target wikipeida-url from the provided queries
  --> sends a request to that url and fetches the comeplete webpage
  --> writes it in a file
"""

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import requests
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

class WikiScraper:
  def __init__(self):
    self.headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
    self.list_urls = []
    self.extra_urls = []
    self.total_urls = 0

  def __call__(self, search_queries, out_file=None, extra_urls=False):
    if out_file is not None:
      for query in tqdm(search_queries, desc="Generating valid urls"):
        target_url = self.build_urls(query)
        self.list_urls.append(target_url)
      
      for url in tqdm(self.list_urls, desc="Scrapping the web-pages\t"):
        out_page = self.scrapper(url)
        with open(out_file, 'a', encoding='utf-8') as f:
          if out_page is not None:
            for paragraph in out_page:
              text = paragraph.get_text()
              f.write(text)
          else: continue
      
      if extra_urls is True:
        for query in tqdm(search_queries, desc="Generating extra urls"):
          extra_urls = self.fetch_extra_urls(query)
          self.extra_urls.append(extra_urls)
        for url in tqdm([item for sublist in self.extra_urls for item in sublist], desc="Scrapping extra urls"):
          extra_output = self.extra_scrape(url)
          with open(out_file, 'a', encoding='utf-8') as f:
            if extra_output is not None:
              for para in extra_output:
                new_text = para.get_text()
                f.write(new_text)
            else: continue
      else:
        pass
      print('\n total fetched urls: ', self.total_urls)
    else:
      raise ValueError('provide a output file')
  
  def build_urls(self, query):
    new_query = '_'.join(query.split(' '))
    wiki_url = f"https://en.wikipedia.org/wiki/{new_query}"
    return wiki_url
  
  def scrapper(self, urls):
    r = requests.get(urls, headers=self.headers)
    if r.status_code == 200:
      soup = bs(r.content, 'html.parser')
      paragraphs = soup.find_all('p')
      self.total_urls += 1
      return paragraphs
    else:
      pass
  
  def fetch_extra_urls(self, query):
    urls = []
    new_query = '_'.join(query.split(' '))
    wiki_url = f"https://en.wikipedia.org/wiki/{new_query}"
    
    r = requests.get(wiki_url, headers=self.headers)
    if r.status_code == 200:
      soup = bs(r.content, 'html.parser')
      links = soup.find_all('a')
      urls.extend([url.get('href') for url in links])
    
    return urls
  
  def extra_scrape(self, url):
    if url.startswith('/'):
      target_url = f"https://en.wikipedia.org{url}"
      r = requests.get(target_url, headers=self.headers)
    else:
      return None
    if r.status_code == 200:
      soup = bs(r.content, 'html.parser')
      paragraphs = soup.find_all('p')
      self.total_urls += 1
      return paragraphs
    else:
        return None