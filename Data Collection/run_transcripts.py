"""
  --> sample script to collect transcripts from youtube videos
"""

import os
from dotenv import load_dotenv
load_dotenv()
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

api_key = os.getenv('yt_key')

from youtube_transcripts import SampleSnippets, TranscriptsCollector
ss = SampleSnippets()
channe_ids = ss()
target_ids = channe_ids[54:68]
out_file = f'../Datasets/transcripts_{len(target_ids)}.txt'

collector = TranscriptsCollector(api_key=api_key)
collector(channel_ids=target_ids, target_file=out_file)