"""
Data Cleaning
=============
This script performs data cleaning and pre-processing
on the song lyrics dataset. The following steps are covered:
- tokenizes lyrics
- lemmatizes words of song lyrics
- re-concatenates words for TFIDF vectorization
- writes updated dataframe to a csv file

To execute this script, run the following command in your terminal:
```
python preprocess.py --filter_by=genre --subset=rock (optional)
```
Due to the large nature of this dataset, lyrics are processed
by genre. Indicate which genre to process using the `genre` arg.

Note: The script automatically writes the updated dataframe to a .csv file.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import argparse
import pandas as pd
import numpy as np
import nltk 

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Song lyric cleaning and pre-processing")
    parser.add_argument('--filter_by', type=str, dest='filter', default='genre')
    parser.add_argument('--subset', type=str, dest='subset', default='rock')

    args = parser.parse_args()

    pd.options.mode.chained_assignment = None

    print("Loading data...")
    data = pd.read_csv("../data/lyrics.csv")
    print("Dropping duplicates and missing values...")
    data = data.dropna()
    data = data.drop_duplicates(subset=['artist', 'song'])

    subset = args.subset
    if args.filter == 'genre':
        subset = args.subset.capitalize()
        if subset == 'Hiphop':
            subset = 'Hip-Hop'
        
    lyrics = data[data[args.filter] == subset]
    print("Number of {:s} songs: ".format(subset), lyrics.shape[0])

    print("Tokenizing lyrics for all {:s} songs...".format(subset))
    lyrics['tokenized_lyrics'] = lyrics['lyrics'].apply(tokenize_lyrics)

    print("Calculating word count...")
    lyrics['word_count'] = lyrics['tokenized_lyrics'].apply(len)

    print("Lemmatizing tokens...")
    lyrics['tokenized_lyrics'] = lyrics['tokenized_lyrics'].apply(lemmatize_lyrics)

    print("Concatenating lyrics...")
    lyrics['processed_lyrics'] = lyrics['tokenized_lyrics'].apply(lambda x: ' '.join(x))

    path = "../data/lyrics_{:s}.csv".format(args.subset.lower())
    lyrics.to_csv(path)
    print("CSV file saved! Check path: {:s}".format(path))
