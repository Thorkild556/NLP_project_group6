import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse
from urllib.parse import parse_qs
from os import getenv
from requests.models import PreparedRequest
from requests import get
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

load_dotenv()

saved_file = 'VideoResults#1.csv'
cols = ("region_code", "video_key", "title")


class ExtractVideos:
    api_key = getenv('API_KEY')
    url = 'https://www.googleapis.com/youtube/v3/videos'
    total_videos_per_region = 1_200 // 6

    def __init__(self):
        self.videos = []
        self.lock = Lock()

    def add_key(self, params):
        return {"key": self.api_key, **params}

    @classmethod
    def get_video_key(cls, p_url):
        parsed_url = urlparse(p_url)
        captured_value = parse_qs(parsed_url.query)['v'][0]
        return captured_value

    def extract(self):
        pd.DataFrame(columns=cols).to_csv(saved_file, columns=cols, index=False)
        with ThreadPoolExecutor(max_workers=5) as executor:
            for region_code in (
                'US',
                'AU', 'GB', 'CA', 'NZ', 'IE'
            ):
                executor.submit(self.fetch_top_videos, region_code)

    def fetch_top_videos(self, region_code):
        code = ''
        for _ in range(self.total_videos_per_region):
            if _ and not code:
                print("STOPPING HERE")
                break

            url = 'https://www.googleapis.com/youtube/v3/videos'
            p = {
                'chart': 'mostPopular',
                'part': 'snippet',
                'regionCode': region_code,
                'maxResults': 50,
                'relevanceLanguage': 'en',
                'fields': 'nextPageToken,items(id,snippet(title))'
            }
            if code:
                p['pageToken'] = code

            params = self.add_key(p)
            req = PreparedRequest()
            req.prepare_url(url, params)

            try:
                resp = get(req.url)
                if not resp.ok:
                    print("ERROR: ", resp.text)
                    break

                _resp = resp.json()
                code = _resp.get('nextPageToken', '')

                rows = [(region_code, v["id"], v["snippet"]["title"]) for v in _resp.get('items', [])]
                with self.lock:
                    pd.DataFrame(rows, columns=cols).to_csv(saved_file, index=False, mode='a', header=False)
                    print(f"✅ {region_code}: +{len(rows)}")
            except Exception as error:
                print(error)
                break

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

if __name__ == "__main__":
    ExtractVideos().extract()
