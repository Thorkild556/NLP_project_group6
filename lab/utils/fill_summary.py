from json import dumps
from os import getenv
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from sqlite3 import connect
import pandas as pd
from lab.utils.ask_llm import OpenAIClient
from lab.utils.batch_utils import batched, fill_prompt, data_folder

class FillSummary(OpenAIClient):
    api_key = getenv('API_KEY')
    url = 'https://www.googleapis.com/youtube/v3/search'
    db_path = data_folder / "saved_videos.db"
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
        try:
            if type(rows) is not pd.DataFrame:
                rows = pd.DataFrame([rows.values.tolist()], columns=list(rows.index))
            keys = dumps(rows.video_key.to_list())
            ask_like_this = fill_prompt(rows, prompt).strip()
            response = self.ask(ask_like_this)
            result_of_prompt = self.parse_response(response)
            self.save_this.append((prompt, ask_like_this, keys, result_of_prompt))
        except Exception as e:
            logger.exception(f"Error processing prompt: {prompt}", exc_info=e)

    def save_results(self):
        pd.DataFrame(self.save_this, columns=("SearchPrompt", "AgentPrompt", "Keys", "Response")).to_sql(
            'PromptResults', self.connection,
            index=False, if_exists='append')
        self.save_this.clear()


#
if __name__ == "__main__":
    with FillSummary() as filler:
        filler.fetch()
