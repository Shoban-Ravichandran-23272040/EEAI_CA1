import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    #max_features will give size of word vec. min_df: if the word appears lesser than 4 times in a row's data, it will be removed
    #similarly if the word is appearing in more than 90% of a single row's data, it will be removed
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

