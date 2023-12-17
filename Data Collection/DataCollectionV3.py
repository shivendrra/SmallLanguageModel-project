from dotenv import load_dotenv
import logging

logging.basicConfig(filename='youtube_fetch.log', level=logging.ERROR)
load_dotenv()

import os
os.chdir('d:/Machine Learning/SLM-Project/Data Collection/')
api_key = os.getenv('yt_secret_key')

from googleapiclient.discovery import build
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
        maxResults = 100, pageToken = next_page_token
      ).execute()

      for item in playlistResult.get('items', []):
        video_response = youtube.videos().list(
          part='snippet',
          id=[item['contentDetails']['videoId']]
        ).execute()
        title = video_response['items'][0]['snippet']['title']
        id = item['contentDetails']['videoId']
        video_data = {'title': title, 'video_id': id}
        videoNo += 1
        vid_json.append(video_data)
      next_page_token = playlistResult.get('nextPageToken')
    
    if not next_page_token:
      break

with open('acs.json', 'w') as file:
  json.dump(vid_json, file, indent=2)

print(f"file written, no of videos fetched were {videoNo}")


# "UCsXVk37bltHxD1rDPwtNM8Q",
#   "UCRcgy6GzDeccI7dkbbBna3Q",
#   "UCmGSJVG3mCRXVOP4yZrU1Dw",
#   "UC415bOPUcGSamy543abLmRA",
#   "UCb_MAhL8Thb3HJ_wPkH3gcw",
#   "UC9RM-iSvTu1uPJb8X5yp3EQ",
#   "UCR1IuLEqb6UEA_zQ81kwXfg",
#   "UCYO_jab_esuFRV4b17AJtAw",
#   "UCA295QVkf9O1RQ8_-s3FVXg",
#   "UCqVEHtQoXHmUCfJ-9smpTSg",
#   "UC4QZ_LsYcvcq7qOsOhpAX4A",
#   "UCLXo7UDZvByw2ixzpQCufnA"