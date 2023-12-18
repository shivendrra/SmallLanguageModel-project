# test code

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
os.chdir('d:/Machine Learning/SLM-Project/Data Collection/')
api_key = os.getenv('openAi_secret_key')

client = OpenAI(api_key=api_key)

prompt = 'can you give me the recipe of cheesecake'
response = client.completions.create(
    model="text-davinci-002",
    prompt=prompt,
    max_tokens=100
)

generated_text = response['choices'][0]['text']
print(generated_text)
