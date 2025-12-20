from numpy import random

class Prepends:
    def __init__(self):
        self.prepends = [
            "Transcript number {a} said:",
            "Transcript {a} was about the following:",
            "This transcript ({a}) summarizes to:",
            "Transcript {a} can be summarized as:",
            "Video {a} covered:",
            "In video transcript {a}, the main points were:",
            "Transcript {a} discussed:",
            "The key takeaways from transcript {a}:",
            "Video transcript {a} explained:",
            "From transcript {a}:",
            "Transcript {a} highlights:",
            "The {a} transcript focused on:",
            "In video {a}, the content was:",
            "Transcript {a} breakdown is:",
            "The main ideas in transcript {a}:",
            "Video transcript {a} summary:",
            "Transcript {a} outlined:",
            "Key points from transcript {a} included:",
            "The {a} video discussed:",
            "Transcript {a} in brief:",
            "Transcript {a} touched on:",
            "The essence of transcript {a} was:",
            "Video {a} explored:",
            "Transcript {a} went over:",
            "What transcript {a} covered:",
            "The gist of transcript {a}:",
            "Video {a} delved into:",
            "Transcript {a} presented:",
            "Core content from transcript {a} was:",
            "Transcript {a} revealed:",
            "Video {a} examined:",
            "The substance of transcript {a}:",
            "Transcript {a} unpacked:",
            "What video {a} was about:",
            "Transcript {a} detailed:",
            "Video {a} addressed:",
            "Video {a} illuminated:"
        ]
        self.preppreps = [
            "However, ",
            "Additionally, ",
            "Furthermore, "
        ]

    def get_random_prepend(self, num):
        rand =  random.randint(0, len(self.prepends))
        if num == 1:
            prep = self.prepends[rand].format(a=num)
        else:
            rand_prepprep = random.randint(0, len(self.preppreps))
            prepprep = self.preppreps[rand_prepprep]
            rando = random.choice(2)
            if rando == 0:
                prep = prepprep + (self.prepends[rand].format(a=num).lower())
            else:
                prep = self.prepends[rand].format(a=num)
        return prep

    def get_n_prepends(self, n):
        preps = []
        for i in range(n):
            preps.append(self.get_random_prepend(i+1))
        return preps

# aim of this file is to prepare the template prefix for the tl;dr clustered response
if __name__ == "__main__":
    prep = Prepends()

    preps = prep.get_n_prepends(4)
    for i in preps:
        print(i)
