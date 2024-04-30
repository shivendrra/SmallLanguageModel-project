"""
  --> contains some sample search queries for the britannica scrapper
"""

class searchQueries:
  def __init__(self):
    self.search_queries = [
      "antarctica",
      "colonization",
      "world war",
      "asia",
      "africa",
      "australia",
      "holocaust",
      "voyages",
      "biological viruses",
      "Martin Luther King Jr",
      "Abraham Lincon",
      "Quarks",
      "Quantum Mechanincs",
      "Biological Viruses",
      "Drugs",
      "Rockets",
      "Physics",
      "Mathematics",
      "nuclear physics",
      "nuclear fusion",
      "CRISPR CAS-9",
      "virginia woolf",
      "cocaine", 
      "marijuana",
      "apollo missions",
      "birds",
      "blogs",
      "journal",
      "Adolf Hitler",
      "Presidents of United States",
      "genders and sexes",
      "journalism",
      "maths theories",
      "matter and particles",
      "discoveries",
      "authoers and writers",
      "poets and novel writers",
      "literature",
      "awards and honors"
    ]
  
  def __call__(self):
    return self.search_queries