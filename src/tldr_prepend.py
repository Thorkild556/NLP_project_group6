from numpy import random

class Prepends:
    def __init__(self):
        self.prepends = [
            "Transcript number {a} said:",
            "Transcript {a} was about the following:",
            "This transcript ({a}) summarizes to:",
            "Transcript {a} can be summarized as:",
            "Video {a} covered:",
            "In transcript {a}, the main points were:",
            "Transcript {a} discussed:",
            "The key takeaways from transcript {a}:",
            "Video transcript {a} explained:",
            "From transcript {a}:",
            "Transcript {a} highlights:",
            "The {a} transcript focused on:",
            "In video {a}, the content was:",
            "Transcript {a} breakdown:",
            "The main ideas in transcript {a}:",
            "Video {a} summary:",
            "Transcript {a} outlined:",
            "Key points from transcript {a}:",
            "The {a} video discussed:",
            "Transcript {a} in brief:",
            "Transcript {a} touched on:",
            "The essence of transcript {a}:",
            "Video {a} explored:",
            "Transcript {a} went over:",
            "What transcript {a} covered:",
            "The gist of transcript {a}:",
            "Video {a} delved into:",
            "Transcript {a} presented:",
            "Core content from transcript {a}:",
            "Transcript {a} revealed:",
            "Video {a} examined:",
            "The substance of transcript {a}:",
            "Transcript {a} unpacked:",
            "What video {a} was about:",
            "Transcript {a} detailed:",
            "The crux of transcript {a}:",
            "Video {a} addressed:",
            "Transcript {a} conveyed:",
            "Central themes in transcript {a}:",
            "Video {a} illuminated:"
        ]
    
    def get_random_prepend(self, num):
        rand =  random.randint(0, len(self.prepends))
        prep = self.prepends[rand].format(a=num)
        return prep


if __name__ == "__main__":
    prep = Prepends()

    print(prep.get_random_prepend(2))
