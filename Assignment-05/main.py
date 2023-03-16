import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from dataset import extract_labels, split_data
import argparse
from features import *
from sklearn.metrics import accuracy_score, confusion_matrix

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
    #print(df.head())
    df['text'] = df['text'].str.lower()
    df = extract_labels(df)

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    try: 
        x_train, y_train, x_test, y_test = ngrams(train, test, args.n_gram) 
        #train, dev, test = split_data(df_features)

    except:
         
        raise KeyError('Acceptable types, word & char')
    
    

    try: 
            clf, model_list = models(args.model) 
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy  = accuracy_score(y_test, y_pred) 
            print(f'Accuracys: {accuracy}')
            print(f'{confusion_matrix(y_test, y_pred)}')
    except:

        raise KeyError(f'The model has not been implemented yet. Please use the ones listed in python main.py --help')
     
