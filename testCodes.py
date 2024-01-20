from googleapiclient.discovery import build
import json
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('yt_secret_key')
os.chdir('D:/Machine Learning/SLM-Project/')
youtube = build('youtube', 'v3', developerKey=api_key)

file_path = 'channelIDs.json'
with open(file_path, 'r') as file:
  channelData = json.load(file)

def fetchVideoUrl(channelId):
    next_page_token = None
    videoIds = []

    while True:
        # fetches the channel's info
        channelRes = youtube.channels().list(
            part='contentDetails', id=channelId
        ).execute()

        # uses the channel info and then fetches the playlist
        if 'items' in channelRes and channelRes['items']:
            playlistId = channelRes['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # uses that playlist info and then fetches the links of uploaded videos
            playlistResult = youtube.playlistItems().list(
                part='contentDetails', playlistId=playlistId,
                maxResults=100, pageToken=next_page_token
            ).execute()

            # append videoIds from the current page
            videoIds.extend([item['contentDetails']['videoId'] for item in playlistResult.get('items', [])])

            next_page_token = playlistResult.get('nextPageToken')

            if not next_page_token:
                break

    return videoIds

# Initialize an empty list to store all videoIds
all_videoIds = []

for channel_id in channelData:
    videoIdUrls = fetchVideoUrl(channel_id)
    all_videoIds.extend(videoIdUrls)

print(all_videoIds)


for channel_id in channelData:
  videoIdUrls = fetchVideoUrl(channel_id)

print(len(videoIdUrls))


import timeit
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv('yt_secret_key')
os.chdir('D:/Machine Learning/SLM-Project')
youtube = build('youtube', 'v3', developerKey=api_key)

start_time = timeit.default_timer()

file_path = 'channelIds.json'
with open(file_path, 'r') as file:
  channelData = json.load(file)

def fetch_url(channelId):
  next_page_token = None
  videos_id = []

  while True:
    channelRes = youtube.channels().list(
      part='contentDetails', id=channelId
    ).execute()

    if 'items' in channelRes and channelRes['items']:
      playlistId = channelRes['items'][0]['contentDetails']['relatedPlaylists']['uploads']

      playlistResult = youtube.playlistItems().list(
        part='contentDetails', playlistId=playlistId,
        maxResults = 100, pageToken = next_page_token
      ).execute()

      videos_id.extend([item['contentDetails']['videoId'] for item in playlistResult.get('items', [])])

      next_page_token = playlistResult.get('nextPageToken')

      if not next_page_token:
        break
  
  return videos_id

all_video_id = []
for channel_id in channelData:
  videoUrls = fetch_url(channel_id)
  all_video_id.extend(videoUrls)

video_fetch_time = timeit.default_timer() - start_time
print(f"video urls fetched in {video_fetch_time} secs")

urls = []
for i in videoUrls:
  videoLink = f"https://www.youtube.com/watch?v={i}"
  urls.append(videoLink)

def convertToJson(videoUrls):
  with open('videoUrls.json', 'w') as file:
    json.dump(videoUrls, file, indent=2)
    print('data written in JSON file successfully!!')

convertToJson(urls)

from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
import logging

logging.basicConfig(filename='youtube_fetch.log', level=logging.ERROR)

def get_captions(videoId):
  try:
    raw_transcripts = []
    videoNo = 0
    for ids in videoId:
      try:
        captions = YouTubeTranscriptApi.get_transcript(
          ids, languages=['en'], preserve_formatting=True
        )
        if captions:
          formatted_captions = [{'text': caption['text']} for caption in captions]
          raw_transcripts.append(formatted_captions)
          videoNo += 1
        else:
          continue
      except TranscriptsDisabled as e:
        print(F"There was an error while getting the captions: {e}")
      except Exception as e:
        logging.error(f"There was some error while fetching the video: {str(e)}")
    print(f"Number of videos with valid captions are: {videoNo}")
    return raw_transcripts
  except Exception as e:
    logging.error(f"There was some error while getting the captions: {str(e)}")

captions_fetch_time = timeit.default_timer() - video_fetch_time
print(f"Captions were fetched in {captions_fetch_time} mins")

captions = get_captions(all_video_id)

with open('captions.txt', 'w', encoding='utf-8') as file:
  for videoCaptions in captions:
    for line in videoCaptions:
      file.write(line['text'] + ' ')

print('Captions file saved successfully!!')
wiriting_time = timeit.default_timer() - captions_fetch_time

print(f'Captions written in {wiriting_time} secs')
print(f"Code exectued in {timeit.default_timer() - start_time} mins")

# # # Flowchart # # #
# Use a for loop to keep the code running untill the last video available
# Use another for loop to use the fetched video_links to get the captions of them
# Use another for loop to write them into a file
# Then the loop main loop runs again until the last video's captions are fetched
# and written in the file

# till thomas