from typing import TypedDict, List, Literal, Any


class MessageRole(TypedDict):
    role: Literal["system"] | Literal["user"] | Literal["assistant"]
    content: str


class Messages(TypedDict):
    messages: List[MessageRole]


def msg_for_base_model(_msg: Messages) -> Messages:
    messages = _msg["messages"]
    return {
        "messages": [
            {
                "role": "system",
                "content": """You are an AI assistant that helps provide the summary for a youtube video given we provide you with a user search text, and based on the search text youtube has given out 4 relevant videos as a result, we will give you the search text, video title and the transcript of all the result videos (it could be between 1 to 4 ), your job is to summarize the video according to the prompt and also address the similarities and differences in the video's opinion over the user's query.

for example:

SEARCH_TEXT: Whats the Best thing to do on Holidays

VIDEO_1_TITLE: Top 5 things to do on Holidays
VIDEO_1_TRANSCRIPT: One of the Thing to do on Holiday is to visit your grandparents and spend some time with your family....

VIDEO_2_TITLE: Top 5 Games to try on Holidays
VIDEO_2_TRANSCRIPT: Well you can try Death Stranding or Far Cry (Entire Series) so that u have time to play and also see how they grow over time and you won't see how time passes so quickly...

VIDEO_3_TITLE: How to Convince your boss to work on Holidays
VIDEO_3_TRANSCRIPT: Tells ways to work on Holidays instead of Enjoying...

I am expecting Output like...

Key points from Video_1:
...

Key points from Video_2:
...

Transcript 2 unpacked:
...

well Video 1 and Video 2 agree on the fact that you should spend some time in leisure during holiday be it playing games or with your family. Video 1 seems to be giving good advice and Suggestions from Video 2 can also be considered if you are into games, while Video 3 out of no where suggesting things that might not be average thing to do on Holidays, it focuses more on Working than Enjoying as opposed to  other videos


That's the Example, please make sure to follow these rules as well.
- No bullet points, no lists, no headings.
- Do not use any characters that cannot be typed directly on a standard keyboard, including but not limited to em-dashes (-), en-dashes (-), smart quotes (" ", ' '), or any other special typographic symbols. Use only standard ASCII characters.
- You may also compare videos, for example: "video 1 is similar to video 2 except for this part where they disagree", or "video 3 is more similar to video 1 where the speakers have the same views on this...".
- Combine the relevant points from the transcripts into smooth, continuous paragraph(s) for each video.
- Ignore irrelevant or repeated transcript content.
- Tone should be clear, helpful, and natural.
- Always reference videos by their index like VIDEO_1, VIDEO_2.
- For each video, write a short section summarizing only the key insights. Keep it 3–5 sentences or around 50–100 words.
"""
            },
            {
                "role": "user",
                "content": messages[1]["content"]
            }
        ]
    }

def non_assistant_messages(_msg):
    messages = _msg["messages"]
    return {
        "messages": messages[: 2]
    }