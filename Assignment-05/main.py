import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from dataset import extract_labels, split_data
import argparse
from features import *
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #path = 'Assignment-05'
    parser.add_argument('--df', dest='df', type=str, required=True,
                    help='Input dataset') 
    parser.add_argument('--model', dest='model', type=str, required=True,
                    help = 'Choose Models from \
                            [SVM, ADA, LR, DT, RF, MNB')
    parser.add_argument('--n_gram', dest='n_gram', type=str, required=True, 
                    help= '--n_gram word \
                        --n_gram char')
    
    args = parser.parse_args() 

    df = pd.read_csv(args.df, sep='\t', header=None, names=['text'], index_col=False)
        
    df['text'] = df['text'].str.lower()
    df = extract_labels(df)
    
    if args.n_gram == "char":
        x_train, y_train, x_dev, y_dev, x_test, y_test = char_ngrams(df) 
        #train, dev, test = split_data(df_features)

    else:
        raise ValueError('Word ngrams has not been implemented yet')
    
    clf = models(args.model) 
    clf.fit(x_train, y_train)
    y_dev_pred = clf.predict(x_dev)
    accuracy  = accuracy_score(y_dev, y_dev_pred) 
    #cross_val_score(clf, x_dev, y_dev, cv=10)

    print(f'Accuracys: {accuracy}')
