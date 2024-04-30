import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

import requests
from bs4 import BeautifulSoup as bs
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class WikiXMLScraper:
  def __init__(self):
    self.headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
    self.processed_urls = set()

  def fetch_url(self, url):
    out_page = self.scrapper(url.strip())
    if out_page is not None:
      text = ''.join([paragraph.get_text() for paragraph in out_page])
      return text
    else:
      return ''

  def scrape(self, url_file, out_file, batch_size=1000):
    with open(url_file, 'r', encoding='utf-8') as f:
      urls = f.readlines()

    urls = list(set([url.strip() for url in urls]))

    with ThreadPoolExecutor(max_workers=40) as executor:
      futures = []
      for i in range(0, len(urls), batch_size):
        urls_batch = urls[i:i+batch_size]
        for url in urls_batch:
          if url not in self.processed_urls:
            future = executor.submit(self.fetch_url, url)
            futures.append(future)
            self.processed_urls.add(url)

      with open(out_file, 'a', encoding='utf-8') as outfile:
        for future in tqdm(futures, desc="Scrapping URLs"):
          text = future.result()
          outfile.write(text)

      print(f'Total fetched URLs: {len(self.processed_urls)}')

  def scrapper(self, url):
    r = requests.get(url, headers=self.headers)
    if r.status_code == 200:
      soup = bs(r.content, 'html.parser')
      paragraphs = soup.find_all('p')
      return paragraphs
    else:
      return None