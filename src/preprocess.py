import re
from nltk.stem import WordNetLemmatizer

# Emoji dictionary
emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire',
    ':(': 'sad', ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry',
    ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked',
    ':-$': 'confused', ':\\': 'annoyed', ':#': 'mute',
    ':X': 'mute', ':^)': 'smile', ':-&': 'confused',
    '$_$': 'greedy', '@@': 'eyeroll', ':-!': 'confused',
    ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
    '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile',
    ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel',
    'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}

def preprocess(textdata):
    processedText = []
    wordLemm = WordNetLemmatizer()

    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(urlPattern, ' URL', tweet)

        for emoji in emojis:
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])

        tweet = re.sub(userPattern, ' USER', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if len(word) > 1:
                word = wordLemm.lemmatize(word)
                tweetwords += word + ' '

        processedText.append(tweetwords)

    return processedText
