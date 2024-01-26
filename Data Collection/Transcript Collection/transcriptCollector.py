import os
import json
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
import logging
import timeit
from tqdm import tqdm  # Import tqdm

start_time = timeit.default_timer()
logging.basicConfig(filename='youtube_fetch.log', level=logging.ERROR)
load_dotenv()

api_key = os.getenv('yt_secret_key')
os.chdir("d:/Machine Learning/SLM-Project/")
youtube = build('youtube', 'v3', developerKey=api_key)

file_path = 'Data Collection/Transcript Collection/channel_ids_snippet.json'
with open(file_path, 'r') as file:
  channelData = json.load(file)

videoNo = 0
for links in channelData:
    next_page_token = None
    videoIds = []

    while True:
        channelRes = youtube.channels().list(
            part='contentDetails,snippet', id=links
        ).execute()
        if 'items' in channelRes and channelRes['items']:
            channel_name = channelRes["items"][0]["snippet"]["title"]
            playlistId = channelRes['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            playlistResult = youtube.playlistItems().list(
                part='contentDetails', playlistId=playlistId,
                maxResults=100, pageToken=next_page_token
            ).execute()

            videoIds.extend([item['contentDetails']['videoId'] for item in playlistResult.get('items', [])])

            next_page_token = playlistResult.get('nextPageToken')

            if not next_page_token:
                break

    with tqdm(total=len(videoIds), desc=f"Fetching '{channel_name}' videos") as pbar:
        for ids in videoIds:
            videoUrl = f"https://www.youtube.com/watch?v={ids}"
            try:
                raw_transcripts = []
                try:
                    captions = YouTubeTranscriptApi.get_transcript(
                        ids, languages=['en'], preserve_formatting=True
                    )
                    if captions:
                        formatted_captions = [{'text': caption['text']} for caption in captions]
                        raw_transcripts.append(formatted_captions)
                        videoNo += 1
                        pbar.update(1)
                    else:
                        continue
                except TranscriptsDisabled as e:
                    continue
                except Exception as e:
                    logging.error(f"There was some error while fetching the video: {str(e)}")
            except Exception as e:
                logging.error(f"There was some error while getting the captions: {str(e)}")

            with open('Data/caption files/new_data.txt', 'a', encoding='utf-8') as file:
                for videoCaptions in raw_transcripts:
                    for line in videoCaptions:
                        file.write(line['text'] + ' ')

print(f"total {videoNo} videos were fetched")
print(f"time taken to execute the code is {(timeit.default_timer() - start_time) / 3600} hrs")
