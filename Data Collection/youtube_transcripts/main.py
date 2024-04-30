"""
  --> uses youtube-v3 api to fetch video ids of each channel
  --> fetches transcripts from those videos, using youtube_transcript_api
    - github repo: https://github.com/jdepoix/youtube-transcript-api/tree/master
    - youtube v3 api: https://developers.google.com/youtube/v3/docs
"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

from googleapiclient.discovery import build
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from tqdm import tqdm
import timeit
import logging
logging.basicConfig(filename='youtube_fetch.log', level=logging.ERROR)

class TranscriptsCollector:
  def __init__(self, api_key=None) -> None:
    self.api_key = api_key
    self.youtube = build('youtube', 'v3', developerKey=self.api_key)
    self.videoNo = 0
    self.skippedVids = 0

  def __call__(self, channel_ids, max_results=100, target_file=None):
    self.target_file = target_file
    start_time = timeit.default_timer()
    for id in channel_ids:
      videoIds, channel_name = self.get_video_ids(id, max_results)
      self.get_transcripts(videoIds, channel_name)
    
    self.total_time = timeit.default_timer() - start_time
    self.generate_summary()
  
  def write_transcripts(self, file, transcripts):
    with open(file, 'a', encoding='utf-8') as f:
      for transcript in transcripts:
        for line in transcript:
          f.write(line['text'] + ' ')

  def get_video_ids(self, channel_id, max_results):
    next_page_token = None
    videoIds = []

    while True:
      channel_res = self.youtube.channels().list(
        part='contentDetails,snippet', id=channel_id
      ).execute()
      if 'items' in channel_res and channel_res['items']:
        channel_name = channel_res["items"][0]["snippet"]["title"]
        playlistId = channel_res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        playlistResult = self.youtube.playlistItems().list(
          part='contentDetails', playlistId=playlistId,
          maxResults=max_results, pageToken=next_page_token
        ).execute()

        videoIds.extend([item['contentDetails']['videoId'] for item in playlistResult.get('items', [])])
        next_page_token = playlistResult.get('nextPageToken')
        if not next_page_token:
          break

    return videoIds, channel_name

  def get_transcripts(self, video_ids, channel_name):
    with tqdm(total=len(video_ids), desc=f"Fetching '{channel_name}' videos") as pbar:
      for ids in video_ids:
        try:
          raw_transcripts = []
          try:
            captions = YouTubeTranscriptApi.get_transcript(ids, languages=['en'], preserve_formatting=True)
            if captions:
              formatted_captions = [{'text': caption['text']} for caption in captions]
              raw_transcripts.append(formatted_captions)
              self.videoNo += 1
              pbar.update(1)
            else:
              pbar.update(1)
              self.skippedVids += 1
              continue
          except TranscriptsDisabled as e:
            pbar.update(1)
            self.skippedVids += 1
            continue
          except Exception as e:
            pbar.update(1)
            self.skippedVids += 1
            continue
        except Exception as e:
          logging.error(f"There was some error while getting the captions: {str(e)}")
        self.write_transcripts(transcripts=raw_transcripts, file=self.target_file)

  def generate_summary(self):
    print("\n")
    print('\t\tSummary\t\t')
    print(f"total time taken: {(self.total_time / 3606):.2f} hrs")
    print(f"total videos that had transcripts: {self.videoNo}")
    print(f"total skipped videos: {self.skippedVids}")