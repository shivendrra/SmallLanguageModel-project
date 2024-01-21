import json
import os
os.chdir('D:/Machine Learning/SLM-Project')

with open('Final Models/Bi-gram/hyperparams.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

import torch

batch_size = params['batch_size']
block_size = params['block_size']
max_iters = params['max_iters']
eval_interval = params['eval_interval']
eval_iters = params['eval_iters']
n_head = params['n_head']
n_embd = params['n_embd']
n_layer = params['n_layer']
dropout = params['dropout']
learning_rate = params['learning_rate']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = """
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
- Hmm. Two clear indicators that
your recruiter is a criminal. I'm sorry, baby bro. I know
how much you love pianos. - I already gave the guy the $500. I sent him a payment digitally. (footsteps knock) - Here, Jordan, why
don't you come with me? I'll show you how to
report the scam to the FTC. Then we'll go over whichever
bank or payment platform you used also. (Loretta sighs) (tense music) - Transfer receipts. This guy's a professional
and he isn't operating alone. Travel in 88. This isn't over. - With so many folks doing
their banking online, it's easier than ever for criminals to try and impersonate banks. In fact, this show was inspired by Steven's encounter with a scammer. - It's true. But instead
of bringing me down, it fueled my passion. - You know, Steven came to me and he said, "Betsy, I know you retired early and moved to an RV in the Salt Flats, but I've got a thrilling idea for a show." And here we are. (both laugh) - You're my twin? - Yeah. (both laugh) - Let's check out how the squad deals with a banking scam. (calm music) - Romance, man. What a bunch of baloney. - Ah, I think it's sweet. - There's nothing sweet
about tender moments and grand gestures, Skip. It's like get your own self-esteem. - Oh, come on, Ace. You telling me that
you've never been in love? - My savings account compromised? No, I didn't authorize
a $12,000 withdrawal. That's my life savings. Of course you're speaking to the real me. My social security number. It's 131. Hey! - Why don't you come with us? We'll explain on the way. (footsteps knock) - Is this the guy? - I've been saving that
money for years, man. I was gonna take my girlfriend to Palermo and hide an engagement ring
inside an arancini ball. (paper rustles) - Palermo is beautiful this time of year. We won't let that dream die. - All right, first thing, Benji, we gotta make sure that your account is actually compromised. Like I know it's only my
second day on the job, but this feels like some funny business. - My life is over. - Benji, focus. Call the number on the
back of your debit card. That's a secure way to see if your account has been compromised. - I called the number. They said my account is secure after all. - You know Benji, a
bank will never call you to ask for personal
"""


# importing training data
file_path = 'Data/captions.txt'
with open(file_path, 'r', encoding='utf-8') as file:
  captions = file.read()

chars = sorted(list(set(captions)))
vocab_size = len(chars)

print(f"list of unique characters in dataset: {''.join(chars)}")
print(f"vocab size is {vocab_size}")

from encoder import EncoderDecoder
ed = EncoderDecoder(n_iters=50, train_data=captions)

input_data = ed.encoder(captions)

# train-test split
n = int(0.9*len(input_data))
train_data = input_data[:n]
val_data = input_data[n:]

train_data = torch.tensor(train_data, dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)


print(train_data[30:105])
print(val_data[:20])

print(ed.decoder(train_data[30:105]))
print(ed.decoder(val_data[:20]))


# from bigram_model import BigramLanguageModel

# model = BigramLanguageModel(n_embd, block_size, dropout, n_head, n_layer, vocab_size)
# m = model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 
# print(sum(p.numel() for p in m.parameters())/1e6, 'Million parameters')

# from train_bigram import train_model
# iter, losses = train_model(m, optimizer, max_iters, eval_interval, eval_iters, train_data, val_data, block_size, batch_size, device)

# print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")