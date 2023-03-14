import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def extract_labels(df):
    
    label_list = []

    for comment in list(df['text'].str.lower()):
        if 'racism' in comment or 'racis' in comment or 'racist' in comment:
            label_list.append('racism') 
        elif 'sex' in comment:
            label_list.append('sexism') 
        else:
            label_list.append('None') 
    # add script to remove it
    return pd.DataFrame(
        {
            'text': df['text'],
            'label': label_list
         
        }
    )

def split_data(df):

    train, test = train_test_split(df,test_size=0.2, random_state=42)
    dev, test = train_test_split(test, test_size=0.5, random_state=42) 

    return train, dev, test

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    path = 'Assignment-05'
    parser.add_argument('-df', dest='df', type=str, required=True,
                    help='Input dataset') 
    args = parser.parse_args() 

    df = pd.read_csv(args.df, sep='\t', header=None, names=['text'], index_col=False)
    
    df['text'] = df['text'].str.lower()
    df = extract_labels(df)

    train, dev, test = split_data(df)
    
    if args.df == 'waseemDataSet.txt':
        if not os.path.exists(path):
            os.mkdir(path)

        train.to_csv(f'Assignment-05/train.csv', index=False)
        dev.to_csv(f'Assignment-05/dev.csv', index=False)
        test.to_csv(f'Assignment-05/test.csv', index=False)
