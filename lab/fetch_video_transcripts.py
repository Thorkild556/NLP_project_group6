import pandas as pd
from lab.fetch_vidoes import saved_file, cols
from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path

class FetchTranscripts:
    file_saved_in = 'SavedVideos.csv'

    def __init__(self):
        self.ytt_api = YouTubeTranscriptApi()
        self.requests = pd.read_csv(saved_file)
        if not Path(self.file_saved_in).exists():
            self.already_fetched = set()
        else:
            self.already_fetched = set(
                pd.read_csv(self.file_saved_in).iloc[:, 1].values.tolist()
            )

    def fetch_videos(self):
        for value in self.requests[~self.requests.video_key.isin(self.already_fetched)].values:
            try:
                value_to_save = self.ytt_api.fetch(value[1]).to_raw_data()
            except Exception as error:
                print("UNABLE TO FETCH", value, error)
                continue
            new_value = list(value)
            print("FETCHED: ", value)
            new_value.append(value_to_save)
            pd.DataFrame(
                [new_value], columns=(*cols, 'raw_transcripts')
            ).to_csv(self.file_saved_in, mode='a', index=False)

if __name__ == "__main__":
    FetchTranscripts().fetch_videos()