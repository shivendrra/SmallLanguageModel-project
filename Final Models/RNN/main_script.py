import numpy as np
import pandas as pd

# # # reading file for training
# # with open('Data/captions.txt', 'r', encoding='utf-8') as file:
# #   dataset = file.read()

# importing data and tokenizing
dataset = """
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

# implementing sub-word level tokenizer
from encoder import EncoderDecoder

tokenize = EncoderDecoder(dataset, n_iters=20)
token_inputs = np.array(tokenize.encoder(dataset))

n = int(0.5*len(token_inputs))
train_data = token_inputs[:n]
val_data = token_inputs[n:]

print(train_data[:20], val_data[:20])

# max_length = max(len(seq) for seq in train_data + val_data)
# train_data_padded = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in train_data])
# val_data_padded = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in val_data])
# data = {'input': train_data_padded.tolist(), 'output': val_data_padded.tolist()}

data = {'input': train_data,
        'output': val_data}
df = pd.DataFrame(data)

# implementing rnn
from mainRNN import SimpleRNN

input_size = 1
hidden_size = 2
output_size = 1
rnn = SimpleRNN(input_size, hidden_size, output_size)

epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(df)):
        x = np.array(df['input'][:i + 1])
        y = np.array([df['output'][:i + 1]])

        # Forward and backward pass
        loss = rnn.backward(x, y, learning_rate)
        total_loss += loss

    # if epoch % 100 == 0:
    print(f'Epoch {epoch}, Loss: {total_loss}')