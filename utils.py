"""
Utils
=====
Contains helper functions for pre-processing and modeling scripts.
"""

import nltk 

def tokenize_lyrics(x, min_length=2):
    """
    Tokenizes a string of lyrics, filters out
    stopwords, strips words that start or end with
    special punctuation (-, ., /).

    Args
    ----
    x : str
        string of lyrics
    min_length : int
        minimum length of word to include in tokenized list

    Returns
    -------
    list
        list of tokenized words
    """
    custom_stopwords = ["'s", "n't", "'m", "'re", "'ll","'ve","...", "ä±", "''", '``','--', "'d", 'el', 'la']
    stopwords = nltk.corpus.stopwords.words('english') + custom_stopwords
    tokens = nltk.word_tokenize(x.lower())
    tokens = [t.strip('-./') for t in tokens]
    tokenized = [t for t in tokens if len(t) > min_length and t.isalpha() and t not in stopwords]
    return tokenized

def lemmatize_lyrics(tokens):
    """
    Lemmatizes tokens using NLTK's lemmatizer tool.
    """
    lemmatized = [nltk.stem.WordNetLemmatizer().lemmatize(t) for t in tokens]
    return lemmatized