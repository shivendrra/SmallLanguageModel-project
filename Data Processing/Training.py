import os
import timeit

os.chdir('D:/Machine Learning/SLM-Project/')
start_time = timeit.default_timer()

# read the data from a file
with open('training_data.txt', 'r', encoding='utf-8') as file:
  captions = file.read()

# Tokenization and lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

tokens = nltk.word_tokenize(captions)
lm = WordNetLemmatizer()
lemmatized_tokens = [lm.lemmatize(token.lower()) for token in tokens if token.isalpha()]

# Convert lemmatized tokens back to text
lemmatized_text = ' '.join(lemmatized_tokens)

# Applying tf-idf
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform([lemmatized_text]).toarray()

print("\nTF-IDF Features:")
print(tfidf_matrix)
print("Feature Names:", tfidf.get_feature_names_out())

# converting the vectors to .csv and then saving it
import pandas as pd

vector_array = pd.DataFrame(tfidf_matrix)
vector_array.to_csv('vector_data.csv')

print('data written to .csv file successfully!!')
print(f"Data vectorized in : {timeit.default_timer() - start_time} mins")