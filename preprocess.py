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
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

from utils import tokenize_lyrics, lemmatize_lyrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data cleaning and feature engineering")
    parser.add_argument('--save_csv', dest='save', default=False, action='store_true')
    parser.add_argument('genre', type=str, dest='genre', default='hiphop')

    args = parser.parse_args()

    pd.options.mode.chained_assignment = None

    print("Loading data...")
    data = pd.read_csv("data/lyrics.csv")
    genre = args.genre.capitalize()
    if args.genre == 'Hiphop':
        genre = 'Hip-Hop'
    lyrics = data[data['genre'] == genre]

    print("Tokenizing lyrics for all songs of genre {:s}...".format(args.genre))
    lyrics['tokenized_lyrics'] = lyrics['lyrics'].apply(tokenize_lyrics)

    print("Lemmatizing tokens...")
    lyrics['tokenized_lyrics'] = lyrics['tokenized_lyrics'].apply(lemmatize_lyrics)

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