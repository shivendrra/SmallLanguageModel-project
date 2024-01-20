data = """Listen up. Demato might not think you're behind this, but make no mistake, Judge Carmichael, I know a sophisticated
scam when I stop one. - Bravo. - So good. So good. Yeah, riveting. - Christina's performance
there is just, her ferocity. - Fantastic. - Welcome to a sneak peek at this season's most extraordinary and
heart pounding new show, "S.A.F.E Squad" starring
Christina Ricci, and also me, and others. Hi, I'm Stev"""

from collections import defaultdict

def vocab_init(data):
  vocab = set(''.join(data))
  return vocab

def get_stats(data):
  pair_freq = defaultdict(int)
  for word in data:
    chars = list(word)
    for i in range(len(chars) - 1):
      pair_freq[(chars[i], chars[i+1])] += 1
  
  return pair_freq

def bpe_main(data):
  