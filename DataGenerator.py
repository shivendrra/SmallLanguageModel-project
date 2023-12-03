# Fetch the links related to some specific youtube channel
import timeit
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv('yt_secret_key')
os.chdir('D:/Machine Learning/SLM-Project/')
youtube = build('youtube', 'v3', developerKey=api_key)

start_time = timeit.default_timer()

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

for channel_id in channelData:
  videoIdUrls = fetchVideoUrl(channel_id)

vidFetchTime = timeit.default_timer()

print(len(videoIdUrls))
print(f"videos fetched in: {vidFetchTime - start_time} secs")

# converting videoIds into videoUrls
urlDict = []
for i in videoIdUrls:
  videoLink = f"https://www.youtube.com/watch?v={i}"
  urlDict.append(videoLink)

def convertToJson(results):
  with open('videoUrls.json', 'w') as outfile:
    json.dump(results, outfile, indent=2)
    print('data written in JSON file successfully')

convertToJson(urlDict)

# get the captions from each video link, if available
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import logging

# Set up logging
logging.basicConfig(filename='youtube_fetch.log', level=logging.ERROR)

# Modify get_captions to store raw transcript data
def get_captions(vidId):
  try:
    raw_transcripts = []
    videoNo = 0
    for ids in vidId:
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
        print(f"Transcripts are disabled for the video: {str(e)}")
      except Exception as e:
        logging.error(f"Error while fetching the videos: {str(e)}")
    print(f"no of videos that had captions were: {videoNo}")
    return raw_transcripts
  except Exception as e:
    logging.error(f"Error in getting captions: {str(e)}")

caption = get_captions(videoIdUrls)

capFetchTime = timeit.default_timer()
print(f"captions fetched in: {capFetchTime - vidFetchTime} secs")

# save those captions in a file, all of them in one
with open('captions.txt', 'w', encoding='utf-8') as file:
  for video_captions in caption:
    for line in video_captions:
      file.write(line['text'] + ' ')
      
print('captions file saved successfully!')
writingTime = timeit.default_timer()
print(f"file written in: {writingTime - capFetchTime} secs")

end_time = timeit.default_timer()
totalTime = (end_time - start_time)
print(f"total time taken: {totalTime} secs")