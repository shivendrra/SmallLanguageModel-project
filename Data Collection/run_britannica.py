""" 
  --> sample script for collecting data from britannica.com
"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
import timeit

from britannica import Scrapper, searchQueries
sq = searchQueries()
queries = sq()

start_time = timeit.default_timer()
outfile = f"../Datasets/britannica_{len(queries)}.txt"
bs = Scrapper(search_queries=queries, max_limit=10)
bs(outfile=outfile)
print(f"total time taken {((timeit.default_timer() - start_time)/60):.2f}mins")