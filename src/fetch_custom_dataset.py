from concurrent.futures import ThreadPoolExecutor
from json import load
from os import getenv
from pathlib import Path
from sqlite3 import connect
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep
from typing import List, TypedDict, Optional
from urllib.parse import parse_qs
from urllib.parse import urlparse
from warnings import deprecated

import pandas as pd
import scrapetube
from dotenv import load_dotenv
from loguru import logger
from requests import get
from requests.models import PreparedRequest
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

load_dotenv()


class TranscriptRecord(TypedDict):
    duration: float
    start: float
    text: str


def raw_data_to_single_text(raw_data: str) -> str:
    """
    multi-line text to single line
    :param raw_data: raw text
    :return: single line text
    """
    texts = "".join(i.replace("\n", " ") + " \n" for i in raw_data)
    return "POST: " + texts


def format_df(title: str, raw_data: str) -> str:
    return "TITLE: " + title + "\n\n" + raw_data_to_single_text(raw_data)


class ExtractVideos:
    """
    In this Pipeline we search for a prompt in YouTube and then fetch its transcripts cautiously
    without getting rate-limited

    we use the set of queries generated (data/queries.txt)

    we have two sequential tasks, each task does things parallely with multiple threads
    1. fetch video results from search query
    2. fetch the transcripts from the videos we collected so far.


    for transcripts, it requires the cookies file collected from logged in browser
    """
    api_key = getenv('API_KEY')
    data = Path(__file__).parent.parent / "data"
    url = 'https://www.googleapis.com/youtube/v3/search'
    db_path = data / "saved_videos.db"
    cols = ("prompt", "video_key", "title")
    videos_per_prompt = 4

    def __init__(self):
        self.connection = connect(str(self.db_path))
        self.lock = Lock()
        self.requests = []
        self.transcript_api = YouTubeTranscriptApi()

    def __enter__(self):
        """
        when the context starts we read from the queries and save them to sqlite for batching
        :return:
        """
        file = self.data / "queries.txt"
        _p = []
        for prompt in file.read_text().split(","):
            _p.append(prompt.strip())
        __p = pd.DataFrame(_p, columns=("Prompt",))
        __p["fetched"] = False
        try:
            __p.to_sql("Prompts", self.connection)
        except Exception:
            ...
        self.requests = pd.read_sql("SELECT Prompt from Prompts where not fetched", self.connection).Prompt.values
        return self

    def add_key(self, params):
        return {"key": self.api_key, **params}

    @classmethod
    def get_video_key(cls, p_url: str) -> str:
        """
        get video key from the url
        :param p_url: video url
        :return: video key
        """
        parsed_url = urlparse(p_url)
        captured_value = parse_qs(parsed_url.query)['v'][0]
        return captured_value

    @deprecated("We would not be using this function anymore, as there's more convenient option: extract_prompts")
    def extract_prompt(self, prompt: str):
        """
        returns top 4 video results from search query and saves the video details (title, video_key)
        from YouTube data API

        :param prompt: YouTube search query
        :return:
        """
        with connect(str(self.db_path)) as conn:
            params = self.add_key({
                'q': prompt,
                'part': 'snippet',
                'maxResults': self.videos_per_prompt,
                'relevanceLanguage': 'en',
                'fields': 'nextPageToken,items(id,snippet(title))'
            })
            req = PreparedRequest()
            req.prepare_url(self.url, params)

            try:
                resp = get(req.url)
                if not resp.ok:
                    logger.warning("Failed to fetch this request: {} due to {}", prompt, resp.text)
                    return

                _resp = resp.json()
                rows = [
                    (prompt, v["id"]['videoId'], v["snippet"]["title"]) for v in _resp.get('items', [])
                ]

                with self.lock:
                    pd.DataFrame(rows, columns=self.cols).to_sql("Videos", conn, if_exists="append", index=False)
                    conn.execute(f'UPDATE Prompts SET fetched = TRUE where prompt = "{prompt}"')
                    conn.commit()

            except Exception as error:
                logger.exception("Failed to fetch the request: {} due to {}", prompt, error)
                return

    def extract_prompt_through_scraping(self, search_query: str):
        """
        we use scrape tube for asking it to return top videos returned by video
        after searching it with the query
        extracting the video details which includes
        title and video id, video_id is required for extracting the transcript
        :param search_query: search query
        :return:
        """
        rows = []
        with connect(str(self.db_path)) as conn:
            try:
                for i, video in enumerate(scrapetube.get_search(search_query)):
                    if i >= self.videos_per_prompt:
                        break
                    title = video.get('title', {}).get('runs', [{}])[0].get('text', 'No title')
                    video_id = video['videoId']
                    rows.append((search_query, video_id, title))

                with self.lock:
                    frame = pd.DataFrame(rows, columns=self.cols)
                    frame["transcript_fetched"] = False
                    frame["has_subs"] = True
                    frame["transcript"] = ''
                    frame.to_sql("Videos", conn, if_exists="append", index=False)
                    logger.info("Fetched {} videos for the prompt: {}", len(rows), search_query)
                    conn.execute(f'UPDATE Prompts SET fetched = TRUE where prompt = "{search_query}"')
                    conn.commit()

            except Exception as error:
                logger.exception("Failed to fetch the request: {} due to {}", search_query, error)
                return

    def extract_prompts(self):
        """
        we then use the batch from sqlite and spawn max. 6 threads at once
        to fetch the video results from search query and for each video we then extract its details and then save it in our sqlite db
        :return:
        """
        with ThreadPoolExecutor(max_workers=6) as executor:
            for request in self.requests:
                executor.submit(self.extract_prompt_through_scraping, request)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        when the context ends we close the sqlite connection
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.connection.close()

    def fetch_transcript_yt_dlp(self, v_k: str) -> List[TranscriptRecord]:
        """
        fetches the transcript for a video given its video key from YouTube DLL Lib.
        :param v_k: video_key
        :return: transcript for video
        """
        logger.info("Fetching the transcripts for video_key: {}", v_k)

        ydl_opts = {
            "skip_download": True,  # we don't need video
            "writesubtitles": True,
            "subtitleslangs": ["en", "en-GB", "en-US"],  # we make sure to fetch only english videos
            "writeautomaticsub": True,
            "cookiefile": str(self.data / "cookies.txt"),
            "subtitlesformat": "json3",
            "sleep_interval_subtitles": 1,
            "quiet": True,
            "ejs_use_node": True,
            "ejs_skip": False,
        }
        if getenv('OXY_PROXY'):
            ydl_opts['proxy'] = getenv('OXY_PROXY')

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ydl_opts["outtmpl"] = str(temp_path / "%(id)s.%(ext)s")

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([v_k])

            json_files = list(temp_path.glob("*.json3"))
            if not json_files:
                raise ValueError("Subtitles are disabled for this video")

            # Load JSON3 and extract plain text
            with open(json_files[0], "r", encoding="utf-8") as f:
                data = load(f)

            transcript = []
            for entry in data['events']:
                start = entry.get("tStartMs", 0) / 1000
                duration = entry.get("dDurationMs", 0) / 1000
                text = "".join(seg.get("utf8", "") for seg in entry.get("segs", [])).replace("\n", " ").strip()
                if text:
                    transcript.append({
                        "text": text,
                        "start": start,
                        "duration": duration
                    })
            return transcript

    def extract_transcripts(self, fallback: Optional[bool] = False, retries: Optional[int] = 3,
                            base_wait: Optional[int] = 2):
        """
        we fetch the transcripts for the video details we collected so far sequentially with thread pool

        :param fallback: if false we use DataAPI (downside: we easily get rate-limited) since we can only fetch approx. 100 videos per day, but we have ~3k videos to fetch.
        :param retries: if failed try thrice (might fail due to rate-limit, if failed we wait more seconds before sending next request) default: 3
        :param base_wait: waits for 2 secs min for every video request
        :return:
        """
        fallback = fallback or bool(getenv("FALLBACK"))

        self.requests = pd.read_sql(
            "SELECT video_key, title FROM Videos WHERE has_subs and NOT transcript_fetched",
            self.connection
        )
        is_not_done = not self.requests.empty

        for idx in self.requests.index:
            v_k = self.requests.loc[idx, "video_key"]
            title = self.requests.loc[idx, "title"]

            for attempt in range(1, retries + 1):
                try:
                    if fallback:
                        value_to_save = self.fetch_transcript_yt_dlp(v_k)
                    else:
                        logger.error(
                            "We are resorting to the transcript youtube transcript API we are not recommended to spam this else we might get our IP banned")
                        # we make sure to fetch only the english videos
                        value_to_save = self.transcript_api.fetch(v_k,
                                                                  languages=["en", "en-GB", "en-US", "en-CA", "en-IN",
                                                                             "en-AU", "en-NZ", "en-ZA",
                                                                             "en-IE"]).to_raw_data()
                    if not value_to_save:
                        logger.warning("#No subtitles available for {}, skipping.", v_k)
                        break

                    frame = pd.DataFrame(value_to_save)
                    frame["video_key"] = v_k

                    formatted_text = format_df(title, frame.text.values)

                    query = """
                            UPDATE Videos
                            SET transcript_fetched = TRUE,
                                transcript         = ?
                            WHERE video_key = ?; \
                            """
                    self.connection.execute(query, (formatted_text, v_k))

                    frame.to_sql(
                        "VideoTranscriptsWithStamps",
                        self.connection,
                        index=False,
                        if_exists="append"
                    )

                    self.connection.commit()
                    logger.info(f"Fetched transcripts for {v_k}")
                    break

                except Exception as e:
                    if "Subtitles are disabled for this video" in str(e):
                        logger.warning("No subtitles available for {}, skipping.", v_k)
                        self.connection.execute("""UPDATE Videos
                                                   SET has_subs = FALSE
                                                   WHERE video_key = ?;""", (v_k,))
                        self.connection.commit()
                        break

                    wait_time = base_wait * (2 ** (attempt - 1))
                    logger.warning(
                        f"[Attempt {attempt}/{retries}] Failed to fetch transcript for '{title}' ({v_k}): {e}"
                    )

                    if attempt < retries:
                        logger.info(f"Retrying in {wait_time}s...")
                        sleep(wait_time)
                    else:
                        logger.error(
                            f"❌ Giving up after {retries} attempts for '{title}' ({v_k})"
                        )
                        break
        return is_not_done


# starts the pipeline
if __name__ == "__main__":
    with ExtractVideos() as extractor:
        extractor.extract_prompts()
        extractor.extract_transcripts(True)
