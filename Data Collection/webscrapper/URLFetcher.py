import os
os.chdir('D:/Machine Learning/SLM-Project/')

import requests
from bs4 import BeautifulSoup

class BritannicaUrls:
  def __init__(self, search_queries, max_limit):
    self.max_limit = max_limit
    self.search_queries = search_queries
    self.headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}

  def build_url(self, query, pageNo):
    formattedQuery = '%20'.join(query.split(' '))
    url = f"https://www.britannica.com/search?query={formattedQuery}&page={pageNo}"
    return url

  def get_target_url(self, targets):
    r = requests.get(targets, headers=self.headers)
    list_url = []

    if r.status_code == 200:
      html_content = r.content
      soup = BeautifulSoup(html_content, 'html.parser')
      fetched_urls = soup.find_all('a', attrs={'class': 'font-weight-bold font-18'})
      list_url.extend([url.get('href') for url in fetched_urls])
      return list_url

    else:
      print(f"skipping this {targets}")

  def generate_urls(self):
    page_urls = []
    for query in self.search_queries:
      pageNo = 1
      for i in range(self.max_limit):
        target_url = self.build_url(query, pageNo)
        pageNo += 1
        new_url = self.get_target_url(target_url)
        page_urls.extend(new_url)
    return page_urls