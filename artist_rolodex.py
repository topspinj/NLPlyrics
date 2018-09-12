"""
Artist Rolodex
==============
Gets artist names from song lyrics dataset.
Fetches a list of artists by: 1) popularity or 2) first letter.
This script is useful if you want to analyze a particular artist
but are not sure if it's in the dataset.

To get most popular artists, run the following command in your terminal:
```
python artist_rolodex.py --popular
```
To get artists that start with the letter 'n', run this command:
```
python artist_rolodex.py --first_letter=n
```
"""
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import argparse
import pandas as pd
import numpy as np
import nltk 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Song lyric cleaning and pre-processing")
    parser.add_argument('--popular', dest='popular', default=False, action='store_true')
    parser.add_argument('--first_letter', type=str, dest='first_letter')

    args = parser.parse_args()

    pd.options.mode.chained_assignment = None

    if len(args.first_letter) > 1:
        raise ValueError("first_letter arg must be 1 character.")

    print("Loading data...")
    data = pd.read_csv("data/lyrics.csv")
    data = data.dropna()

    if args.first_letter:
        data['artist_first_letter'] = data['artist'].apply(lambda x: x[0])
        subset_data = data[data['artist_first_letter'] == args.first_letter]
        artists = subset_data['artist'].unique()
        print("Artists with first letter {:s}...".format(args.first_letter))
    if args.popular:
        most_popular = data['artist'].value_counts().reset_index()
        most_popular.columns = ['artist', 'count']
        artists = most_popular.head(20)['artist']
        print("Top 20 most popular artists...")
    print(list(artists))