""" 
--> Bhav nhi deta main iss code ko
--> kuch to krta h, bss malum nhi
"""

from bs4 import BeautifulSoup
import requests
import json

file_path = './search_queries.json'
with open(file_path, 'r', encoding='utf-8') as file:
  url_dict = json.load(file)

def fetch_url(snippet):
  url = f"https://www.britannica.com{snippet}"
  headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"} 
  r = requests.get(url=url, headers=headers)
  return r

def scrapper(content):
  soup = BeautifulSoup(content, 'html5lib')
  links = []
  for link in soup.find_all('a', href=True):
    links.append(link['href'])

  paragraph_data = soup.find_all('p')
  return paragraph_data, links

def main():
  for query_data in url_dict:
    query = query_data['query']
    link = query_data['links']

    print(f"scarpping html for '{query}' now")
    for link_list in link:
      for links in link_list:
        html_content = fetch_url(links)
        scrapped_html, extra_urls = scrapper(html_content.content)
        print(scrapped_html)
        # with open(f'codes/html content/{query}.txt', 'w', encoding='utf-8') as outfile:
        #   outfile.write(scrapped_html)

if __name__ == "__main__":
  main()