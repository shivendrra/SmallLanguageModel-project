"""
  --> sample script to collect data from wikipedia.com
"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
import timeit

from wikipedia import WikiQueries, WikiScraper, WikiXMLScraper
queries = WikiQueries()
scrape = WikiScraper()

queries = queries()
output_file = f'../Datasets/wiki_{len(queries)}.txt'
scrape(out_file=output_file, search_queries=queries, extra_urls=True)