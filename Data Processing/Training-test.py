import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
text_data = """
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
information, or even an OTP. - OTP? - One-time passcode. The next time it happens, you'll know it's a criminal on the phone. - I really appreciate it. You said Palermo was
beautiful this time of year. You been? - You said you were going to propose. (Loretta scoffs) Take it from me, kid. Don't wait. - [Betsy] You all right? - Yeah. Next, you are going to
see the squad dealing with their toughest case
yet, a government scam. - Now, I know a lot of things famously, but I had no idea how prevalent
these kinds of scammers are. Some of the most common
government scams include Covid scams, social security scams, and as you're about to see, IRS scams. (cheery music) - Yes, Mrs Velazquez, I got the note about
the pistachio allergy, and I have taken that into
account for your son's wedding. Your son's third birthday party. Yes, drop off is at the
pavilion at 8:00 AM tomorrow. (Sandra winces) (phone rings) Hello again, Mrs Velazquez. This is she. She said that she was a
representative from the IRS and that I was $6,000
behind on my tax payments. She said- - [Scammer] Now, if
you're unable to wire us that late payment within
48 hours, Ms Gerardi, we will have to send a
sheriff to your home. - The thing is, I run multiple businesses. I think I file my taxes right, but- - It is true, Sandra, that
on rare occasions the IRS will call you at home, but after they've attempted to contact you by mail or other means. - How did the rest of the phone call go? - I asked her to put me onto her manager, who's this older man. - [Scammer] The failure to
pay penalty is one half of 1% for each month. I asked him about my bill
and he transferred me to some southern lady in
the mailing department. - [Scammer] Nah sugar, that's form 2210. - And then, I was talking
to someone in the mail room. He seemed really young.
I probably sound crazy. - Trust me, Sandra. That feeling is what
these criminals prey upon. Their tactics are very complex. - I dunno. Maybe I should
just wire them the money. - Whoa, whoa, whoa, whoa.
"""

# Tokenization and lemmatization
tokens = word_tokenize(text_data)
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]

# Convert lemmatized tokens back to text
lemmatized_text = ' '.join(lemmatized_tokens)

# Applying tf-idf
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([lemmatized_text]).toarray()

print("\nTF-IDF Features:")
print(tfidf_matrix)
print("Feature Names:", tfidf_vectorizer.get_feature_names_out())