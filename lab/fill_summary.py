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

    return f"""
Your task is to read the YouTube video transcripts (1–4 videos) and produce a coherent response that directly answers the user's prompt in a video-wise format. Each video should have its own section with insights, and you may naturally reference other videos for similarities or differences, always citing the video by its title.

IMPORTANT RULES:
- No bullet points, no lists, no headings.
- Do not use any characters that cannot be typed directly on a standard keyboard, including but not limited to em-dashes (-), en-dashes (-), smart quotes (" ", ' '), or any other special typographic symbols. Use only standard ASCII characters.
- You may say "one of the videos explains…", "another video recommends…", or "another video is against ... which was recommended by video titled: [title_of_video_1]".
- You may also compare videos, for example: "video 1 is similar to video 2 except for this part where they disagree", or "video 3 is more similar to video 1 where the speakers have the same views on this...".
- Combine the relevant points from the transcripts into smooth, continuous paragraph(s) for each video.
- Ignore irrelevant or repeated transcript content.
- Tone should be clear, helpful, and natural.
- Always reference videos by their index like VIDEO_1, VIDEO_2.
- For each video, write a short section summarizing only the key insights. Keep it 3–5 sentences or around 50–100 words. 

EXAMPLE (for style only):

USER PROMPT: "How to stop puppy biting instantly"

EXAMPLE (for style only):

USER PROMPT: "How to stop puppy biting instantly"

EXPECTED STYLE / OUTPUT:

[VIDEO_1]: In How To Stop Puppy Biting Instantly, the video emphasizes redirecting a puppy's attention whenever it bites and rewarding calm behavior with treats. The video also recommends using consistent verbal cues like "no" to reinforce boundaries. This aligns with How to Train a Puppy NOT to BITE, which focuses on positive reinforcement and patience during play. While both videos stress teaching the puppy proper behavior, How To Stop Puppy Biting Instantly highlights immediate redirection more strongly.

[VIDEO_2]: How to Train a Puppy NOT to BITE adds that ignoring playful nips can help the puppy learn self-control, which contrasts slightly with How To Stop Puppy Biting Instantly where redirection is prioritized. The video also suggests observing the puppy's body language to anticipate biting. Comparing the videos, it becomes clear that combining redirection, consistent commands, and rewarding calm behavior is the most effective approach to stopping puppy biting.

END OF EXAMPLE
-----------------------------------------

ACTUAL USER PROMPT: {prompt}

VIDEO TRANSCRIPTS:
{videos_text}
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

    def fetch(self):
        q = "SELECT prompt, video_key, title, transcript FROM Videos WHERE transcript_fetched and transcript is NOT NULL"

        try:
            self.requests = pd.read_sql(
                q,
                self.connection, index_col="prompt"
            )
        except:
            self.requests = pd.read_sql(
                """
SELECT v.prompt, v.video_key, v.title, v.transcript 
FROM Videos
left join PromptResults p on v.prompt = p.prompt
WHERE p.prompt is NULL and transcript_fetched and transcript is NOT NULL"
                """,
                self.connection, index_col="prompt"
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            for request in {'how to make sourdough bread from scratch'} or set(self.requests.index):
                executor.submit(self.ask_llm, self.requests.loc[request], request)

        self.save_results()

    def ask_llm(self, rows, prompt):
        keys = rows.video_key.to_list()
        ask_like_this = fill_prompt(rows, prompt).strip()
        result_of_prompt = ...
        print(ask_like_this)

        # self.save_this.append((prompt, keys, result_of_prompt))

    def save_results(self):
        pd.DataFrame(self.save_this, columns=("Prompt", "Keys", "Response")).to_sql('PromptResults', self.connection, index=False, if_exists='append')


if __name__ == "__main__":
    with FillSummary() as filler:
        filler.fetch()