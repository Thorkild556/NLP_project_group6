from os import getenv
from requests import get
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from sqlite3 import connect
from pathlib import Path
import pandas as pd
from itertools import islice

def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def fill_prompt(prompt, title, transcript: str):
    return f"""
Please generate the TL;DR for the below youtube video's details

PROMPT: {prompt}

TITLE: {title} 

TRANSCRIPT: {transcript}

"""

class FillSummary:
    api_key = getenv('API_KEY')
    data = Path(__file__).parent.parent / "data"
    url = 'https://www.googleapis.com/youtube/v3/search'
    db_path = data / "saved_videos.db"
    videos_per_prompt = 4

    def __init__(self):
        self.connection = connect(str(self.db_path))
        self.requests = []
        self.save_this = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def __enter__(self):
        return self

    def fix_col(self):
        try:
            self.connection.execute(
                """ALTER TABLE Videos ADD COLUMN tl_dr TEXT DEFAULT ''"""
            )
            self.connection.commit()
        except:
            ...

    def fetch(self):
        self.fix_col()
        self.requests = pd.read_sql(
            "SELECT prompt, video_key, title, transcript FROM Videos WHERE transcript_fetched and transcript is NOT NULL",
            self.connection
        )
        for batch in batched(self.requests.iterrows(), 10):
            with ThreadPoolExecutor(max_workers=10) as executor:
                for _, sub_batch in batch:
                    executor.submit(self.ask_llm, *sub_batch.values)
            self.save_results()

    def ask_llm(self, prompt, video_key, title, transcript):
        ask_like_this = fill_prompt(prompt, title, transcript)
        result_of_prompt = ...

        self.save_this.append((video_key, result_of_prompt))

    def save_results(self):
        print(self.save_this)
        self.save_this.clear()


if __name__ == "__main__":
    with FillSummary() as filler:
        filler.fetch()