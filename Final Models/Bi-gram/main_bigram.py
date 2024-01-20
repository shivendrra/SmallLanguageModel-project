
data = """Listen up. Demato might not think you're behind this, but make no mistake, Judge Carmichael, I know a sophisticated
scam when I stop one. - Bravo. - So good. So good. Yeah, riveting. - Christina's performance
there is just, her ferocity. - Fantastic. - Welcome to a sneak peek at this season's most extraordinary and
heart pounding new show, "S.A.F.E Squad" starring
Christina Ricci, and also me, and others. Hi, I'm Stev"""

n_merges = 10
tokenizer = SubwordTokenizer(n_merges)
final_vocab = tokenizer.apply_bpe(data)
tokens = tokenizer.tokenize_data(data, final_vocab)
de_token = tokenizer.detokenize_data(tokens)

print(tokens)
print(de_token)