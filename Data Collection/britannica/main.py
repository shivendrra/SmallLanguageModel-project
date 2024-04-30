import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

from .URLFetcher import BritannicaUrls as bu
import requests
import re
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

class Scrapper(bu):
  def __init__(self, search_queries=None, max_limit=10):
    super().__init__(search_queries=search_queries, max_limit=max_limit)
    self.search_queries = search_queries
    self.max_limit = max_limit
    self.headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
  
  def __call__(self, outfile):
    url_snippets = self.scraper(self.search_queries)
    with tqdm(total=len(url_snippets), desc="Scrapping in progress: ") as pbar:
      for snippets in url_snippets:
        page = self.text_extractor(snippets)
        if outfile is not None:
          if page is not None:
            with open(outfile, 'a', encoding='utf-8') as f:
              f.write(page)
          else: continue
          pbar.update(1)
        else:
          raise ValueError("Provide valid outfile with path")
    print("Data collected and saved successfully!!")

  def text_extractor(self, url_snippet):
    target_url = f"https://britannica.com{url_snippet}"
    r = requests.get(target_url, headers=self.headers)

    if r.status_code == 200:
      soup = bs(r.content, 'html.parser')
      paragraphs = soup.find_all('p')

      page = '\n'.join([p.get_text() for p in paragraphs if "Our editors will review what you’ve submitted and determine whether to revise the article." not in p.get_text()])
      page = re.sub('&\w+;','',page)

      return page
  
  def scraper(self, query):
    scrapped = bu(query, max_limit=self.max_limit)
    with tqdm(total=len(self.search_queries) * self.max_limit, desc="Generating URL snippets: ") as pbar:
      url_snippets = scrapped.generate_urls(progress_bar=pbar)
    
    return url_snippets