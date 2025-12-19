from itertools import islice
import pandas as pd
from pathlib import Path


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
    .replace("TITLE", f"TITLE_OF_VIDEO_RESULT_{index + 1}")
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


data_folder = Path(__file__).parent.parent.parent / "data"