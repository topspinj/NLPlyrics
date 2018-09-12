"""
Data Cleaning
=============
This script performs data cleaning and pre-processing
on the song lyrics dataset. The following steps are covered:
- tokenizes lyrics
- lemmatizes words of song lyrics
- creates a TFIDF sparse csr matrix for topic modeling

To execute this script, run the follow command in your terminal:
```
python preprocess.py --genre=hiphop --save_csv (optional)
```
Due to the large nature of this dataset, lyrics are processed
by genre. Indicate which genre to process using the `genre` arg.

The script automatically writes the TFIDF matrix to a .npz file.
However, if you would like to also save the updated dataframe with
additional column `tokenized_lyrics`, then add the `save_csv` flag
to the command above. 
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
        description="Data cleaning and feature engineering")
    parser.add_argument('--save_csv', dest='save', default=False, action='store_true')
    parser.add_argument('--genre', type=str, dest='genre', default='hiphop')

    args = parser.parse_args()

    pd.options.mode.chained_assignment = None

    print("Loading data...")
    data = pd.read_csv("data/lyrics.csv")
    data = data.dropna()
    genre = args.genre.capitalize()
    if genre == 'Hiphop':
        genre = 'Hip-Hop'
    lyrics = data[data['genre'] == genre]
    print("Number of {:s} songs: ".format(genre), lyrics.shape[0])

    print("Tokenizing lyrics for all songs of genre {:s}...".format(genre))
    lyrics['tokenized_lyrics'] = lyrics['lyrics'].apply(tokenize_lyrics)
    print(lyrics['tokenized_lyrics'][249])

    print("Calculating word count...")
    lyrics['word_count'] = lyrics['tokenized_lyrics'].apply(len)

    print("Lemmatizing tokens...")
    lyrics['tokenized_lyrics'] = lyrics['tokenized_lyrics'].apply(lemmatize_lyrics)
    print(lyrics['tokenized_lyrics'][249])

    print("Concatenating lyrics...")
    processed_lyrics = lyrics['tokenized_lyrics'].apply(lambda x: ' '.join(x))

    print("Performing TFIDF vectorization...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(processed_lyrics)
    save_npz('data/tfidf_{:s}.npz'.format(args.genre), X)
    print("TFIDF matrix saved! Check data/ directory.")

    if args.save is True:
        path = "data/lyrics_{:s}.csv".format(args.genre)
        lyrics.to_csv(path)
        print("CSV file saved! Check path: {:s}".format(path))
