"""
Topic Modeling
==============
- creates a TFIDF sparse csr matrix for topic modeling
- extracts topics using LDA
- extracts topics using NMF
"""

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import argparse
import pandas as pd
import numpy as np
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def get_topics(k_topics, feature_names, model, top_n_words=20):
    """
    Gets k topics from a LDA/NMF model and prints the top n words.
    """
    for i in range(0,k_topics):
        topic = pd.DataFrame(data={'word':feature_names, 'weight':model.components_[i]})
        sorted_topic = topic.sort_values('weight', ascending=False).head(top_n_words)
        print("Topic %s:" % i, ' '.join(sorted_topic['word']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Topic Modeling with LDA and NMF")
    parser.add_argument('--name', type=str, dest='name')
    parser.add_argument('--k_topics', type=int, dest='k_topics', default=10)
    parser.add_argument('--n_top_words', type=int, dest='n_top_words', default=10)

    args = parser.parse_args()

    pd.options.mode.chained_assignment = None

    print("Loading data...")
    data = pd.read_csv("data/lyrics_{:s}.csv".format(args.name))
    data = data.dropna()
    
    print("Performing TFIDF vectorization...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(data['processed_lyrics'])
    features = tfidf.get_feature_names()
    print("TFIDF matrix dimensions for {:s}:".format(args.name), X.shape)

    print("{:d} topics using NMF:".format(args.k_topics))
    nmf = NMF(n_components=args.k_topics)
    nmf.fit(X)
    get_topics(args.k_topics, features, nmf, args.n_top_words)