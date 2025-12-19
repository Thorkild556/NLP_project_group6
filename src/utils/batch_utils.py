from itertools import islice
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Generator, Any


def batched(iterable: List, n: int, *, strict: Optional[bool] = False) -> Generator[Tuple[Any]]:
    """
    batches the iterable return list of tuples with batched items
    :param iterable: list of items
    :param n: max no. of items in a batch
    :param strict: if strict makes sures every batch has n elements but raises if it couldn't
    :return: batch items
    """
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def generate_for_row(index: int, row: Any) -> str:
    """
    for every row, since we stored in such a way that the title is referred as TITLE and transcripts as POST
    we would be replacing them as per the prompt instructions
    :param index: index of the video result
    :param row: video row
    :return: prompt for this video
    """
    return f"""
{
    row.transcript
    .replace("TITLE", f"TITLE_OF_VIDEO_RESULT_{index + 1}")
    .replace("POST", f"TRANSCRIPT_OF_VIDEO_RESULT_{index + 1}")
    }
"""


def fill_prompt(rows: pd.DataFrame, search_query: str) -> str:
    """
    we generate the prompt for every video result and then combine to get a final prompt.
    :param rows: video results for a YouTube Search query
    :param search_query: YouTube Search Query
    :return: final prompt for the search query
    """
    videos_text = "\n".join(
        generate_for_row(i, row) for i, row in enumerate(rows.itertuples(index=False))
    )

    return f"""USER PROMPT: {search_query}
VIDEO TRANSCRIPTS :-
{videos_text}
""".strip()


data_folder = Path(__file__).parent.parent.parent / "data"
