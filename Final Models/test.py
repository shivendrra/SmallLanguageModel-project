import os
os.chdir('D:/Machine Learning/SLM-Project/')


input_data = """
- Listen up. Demato might not think you're behind this, but make no mistake, Judge Carmichael, I know a sophisticated
scam when I stop one. - Bravo. - So good. So good. Yeah, riveting. - Christina's performance
there is just, her ferocity. - Fantastic. - Welcome to a sneak peek at this season's most extraordinary and
heart pounding new show, "S.A.F.E Squad" starring
Christina Ricci, and also me, and others. Hi, I'm Steven Wu. - And I'm Betsy Chate. Our show follows a trio of investigators dedicated to stopping
criminals in their tracks. Now, imposter scams can
involve anyone who's pretending to be someone else in order to trick you into providing personal information or even sending money. - They're real and they're
not going away anytime soon. - We have so much to show you, so let's dive into a few
clips from our pilot episode of "S.A.F.E. Squad". (tense music) - Uh huh. Uh huh. So the message came from
the job posting site. Listen, why don't you come in today? We'll take a look, see what we can do. All right, Jordan. See you soon. Call the boss. Ask her to come in. - I know it's only my first day, but it sounds like a pretty
cut and dry job scam. Can't you and I just help this guy? - It's not that simple, rookie. He's the boss's baby brother. - Okay, baby bro. Walk
me through what happened. - So I'm on this job listing site and I get a message from a recruiter for a small shipping company. Small shipping company. It said I would be perfect for a role in their operations department. Company was called Travel in 88. Travel in 88. We specialize in your
unique piano shipping needs. The recruiter said, all I
needed to do was send $500 to cover mandatory software
training and job was mine. - Baby bro, come on. - If this website is a front, I mean, it's a pretty good one. - "I wouldn't trust anyone
else with my orwolu upright." - Ormolu. Have we compared the website's URL with the domain name of the
recruiter's email address? - They don't match up. I even tried the hotline
number at the bottom. Dead end.
"""

# with open('Data/captions.txt', 'r', encoding='utf-8') as file:
#   input_data = file.read()

from sub_wordTokenizer import SubwordTokenizer

vocab = sorted(list(set(input_data)))
chars = ''.join(sorted(list(set(input_data))))
n_merges = 10
tokenizer = SubwordTokenizer(n_merges)
tokens = tokenizer.tokenize_data(input_data, vocab)

stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

import numpy as np

input_data = np.array(encode(tokens))
print(input_data.shape)