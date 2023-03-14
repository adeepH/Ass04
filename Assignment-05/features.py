import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from dataset import extract_labels, split_data
import argparse
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def char_ngrams(df):

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    dev, test = train_test_split(test, test_size=0.5, random_state=42) 

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(5,5))
    x_train = vectorizer.fit_transform(train['text'])
    x_dev = vectorizer.transform(dev['text'])
    x_test = vectorizer.transform(test['text'])
    y_train, y_dev, y_test = train['label'], dev['label'], test['label']    

    return x_train, y_train, x_dev, y_dev, x_test, y_test

def models(args):

    model_dict={
        'SVM': LinearSVC(),
        'MNB': MultinomialNB(),
        'LR' : LogisticRegression(),
        'DT': DecisionTreeClassifier(),
        'ADA': AdaBoostClassifier(),
        'RF': RandomForestClassifier(),
    }

    return model_dict[args]
