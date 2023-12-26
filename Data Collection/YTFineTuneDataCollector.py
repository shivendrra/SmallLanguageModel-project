from dotenv import load_dotenv
import logging

logging.basicConfig(filename='youtube_fetch.log', level=logging.ERROR)
load_dotenv()

import timeit
start_time = timeit.default_timer()

import os
os.chdir('d:/Machine Learning/SLM-Project/Data Collection/')
api_key = os.getenv('yt_secret_key')

from googleapiclient.discovery import build
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
youtube = build('youtube', 'v3', developerKey=api_key)

import json
file_path = 'channelIDs.json'
with open(file_path, 'r') as file:
  channelData = json.load(file)
    
videoNo = 0
for links in channelData:
  next_page_token = None
  vid_json = []

  while True:
    channelRes = youtube.channels().list(
      part='contentDetails', id=links
    ).execute()
    
    if 'items' in channelRes and channelRes['items']:

      playlistId = channelRes['items'][0]['contentDetails']['relatedPlaylists']['uploads']

      playlistResult = youtube.playlistItems().list(
        part='contentDetails', playlistId=playlistId,
        maxResults = 50, pageToken = next_page_token
      ).execute()

      for item in playlistResult.get('items', []):
        video_response = youtube.videos().list(
          part='snippet',
          id=[item['contentDetails']['videoId']]
        ).execute()
        
        title = video_response['items'][0]['snippet']['title']
        id = item['contentDetails']['videoId']
        
        videoUrl = f"https://www.youtube.com/watch?v={id}"
        try:
          captions = YouTubeTranscriptApi.get_transcript(
            id, languages=['en'], preserve_formatting=True
          )
          if captions:
            formatted_captions = [{caption['text']} for caption in captions]
            raw_transcripts = ' '.join(caption.pop() for caption in formatted_captions)
          else:
            continue
        except TranscriptsDisabled as e:
          print(F"There was an error while getting the captions: {e}")
        except Exception as e:
          logging.error(f"There was some error while fetching the video: {str(e)}")

        video_data = {'title': title, 'video_id': id, 'captions': str(raw_transcripts)}
        videoNo += 1
        vid_json.append(video_data)
      next_page_token = playlistResult.get('nextPageToken')
    
    if not next_page_token:
      break

with open('acs.json', 'w') as file:
  json.dump(vid_json, file, indent=2)

end_time = timeit.default_timer()
print(f"file written, no of videos fetched were {videoNo}")
print(f"time taken to fetch the data {(end_time - start_time) /60} mins")