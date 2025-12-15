from json import dumps
from os import getenv
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from sqlite3 import connect
from pathlib import Path
import pandas as pd
from itertools import islice

from lab.ask_llm import OpenAIClient


def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def generate_for_row(index, row):
    return f"""
{
    row.transcript
    .replace("TITLE:", f"TITLE_OF_VIDEO_RESULT_{index + 1}")
    .replace("POST", f"TRANSCRIPT_OF_VIDEO_RESULT_{index + 1}")
    }
"""


def fill_prompt(rows: pd.DataFrame, prompt: str):
    videos_text = "\n".join(
        generate_for_row(i, row) for i, row in enumerate(rows.itertuples(index=False))
    )

    return f"""USER PROMPT: {prompt}
VIDEO TRANSCRIPTS :-
{videos_text}
""".strip()


class FillSummary(OpenAIClient):
    api_key = getenv('API_KEY')
    data = Path(__file__).parent.parent / "data"
    url = 'https://www.googleapis.com/youtube/v3/search'
    db_path = data / "saved_videos.db"
    videos_per_prompt = 4

    def __init__(self):
        super().__init__()
        self.connection = connect(str(self.db_path))
        self.requests = []
        self.save_this = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def __enter__(self):
        return self

    def fetch(self):
        q = "SELECT prompt, video_key, title, transcript FROM Videos WHERE transcript_fetched and transcript is NOT NULL"
        try:
            self.requests = pd.read_sql(
                """
                SELECT v.prompt, v.video_key, v.title, v.transcript
                FROM Videos v
                where v.prompt not in (select "SearchPrompt" from PromptResults)
                  and v.transcript_fetched
                  and v.transcript is NOT NULL
                """,
                self.connection, index_col="prompt"
            )
        except:
            logger.warning("Resetting to default query for fetching the pending requests")
            self.requests = pd.read_sql(
                q,
                self.connection, index_col="prompt"
            )

        results = set(self.requests.index)
        logger.info(f"Total requests: {len(results)}")
        for idx, request in enumerate(batched(results, 3)):
            logger.info(f"Processing batch {idx + 1} of {len(results) // 3 + 1} - batch of {len(request)}")
            with ThreadPoolExecutor(max_workers=3) as executor:
                for r in request:
                    executor.submit(self.ask_llm, self.requests.loc[r], r)
            logger.info(f"Finished batch: {idx + 1}")
            sleep(1.5)
            self.save_results()

    def ask_llm(self, rows, prompt):
        keys = dumps(rows.video_key.to_list())
        ask_like_this = fill_prompt(rows, prompt).strip()
        result_of_prompt = self.parse_response(self.ask(ask_like_this))
        self.save_this.append((prompt, ask_like_this, keys, result_of_prompt))

    def save_results(self):
        pd.DataFrame(self.save_this, columns=("SearchPrompt", "AgentPrompt", "Keys", "Response")).to_sql(
            'PromptResults', self.connection,
            index=False, if_exists='append')
        self.save_this.clear()


#
if __name__ == "__main__":
    with FillSummary() as filler:
        filler.fetch()
