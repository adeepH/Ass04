import pandas as pd
import numpy as np
import re
import argparse
from sklearn.model_selection import train_test_split

def return_bool(row, lex:str): 

    if lex.lower() in row.str.lower():
        print(lex)
        print(row.str.lower())
        print("True")
        return 1
    else: 
        return 0
        

def extract_labels(df):

    label_list = []

    for comment in list(df['text'].str.lower()):
        if 'racism' in comment or 'racis' in comment or 'racist' in comment:
            label_list.append('racism')
        elif 'sex' in comment:
            label_list.append('sexism')
        else:
            label_list.append('None')
    
    df['label'] = label_list
    return df
    # add script to remove it
    #return pd.DataFrame(
    #    {
    #        'text': df['text'],
    #        'label': label_list

    #    }
    #)

def generate_features(df, lexicon):

    lexicons = lexicon['lexicons']

    for lexicon in list(lexicons):
        #print(lexicon) 
        #lex = df.apply(return_bool, axis=1, args=(lexicon,)) 
        df[lexicon] = df['text'].str.contains(lexicon.lower())
        #print(f"generated a boolean feature vector for {lexicon}")
        #df = pd.concat([df, lex], axis=1) 
    bool_dict = {True: 1, False: 0} 
    df = df.replace(bool_dict)

    #
    return df

def split_data(df):

    train, test = train_test_split(df,test_size=0.2, random_state=42)
    dev, test = train_test_split(test, test_size=0.5, random_state=42) 

    return train, dev, test

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument('-df', dest='df', type=str, required=True,
                    help='Input dataset')

    parser.add_argument('-lex', dest='lexicon',type=str, required=True,
                    help='Lexicon dataset')
    
    args = parser.parse_args() 

    df = pd.read_csv(args.df, sep='\t', header=None, names=['text'], index_col=False)
    df['text'] = df['text'].str.lower()
    
    lexicon = pd.read_csv(args.lexicon, header=None, names=['lexicons'], index_col=False)
    lexicon['lexicons'] = lexicon['lexicons'].str.lower()
    #print(list(lexicon['lexicons'])[0])

    df_transformed = generate_features(df, lexicon)
    print(df_transformed.head(5))
    df_with_labels = extract_labels(df_transformed)
    print(df_with_labels.head(5))
    features = df_with_labels.drop(columns=['text'])
    train, dev, test = split_data(features)

    if args.lexicon == "data/Wiegand_lexicon/hate_lexicon_wiegand.txt":

        features.to_csv(
            r'data/Wiegand_lexicon/hate_lexicon_wiegand_df.csv', index=False)
        print("writing train data")
        train.to_csv(r'data/Wiegand_lexicon/train_features_lexicon_wiegand.csv', index=False, header=False)
        print("Writing test data")
        test.to_csv(r'data/Wiegand_lexicon/test_features_lexicon_wiegand.csv', index=False, header=False)
        print("writing dev data")
        dev.to_csv(r'data/Wiegand_lexicon/dev_features_lexicon_wiegand.csv', index=False, header=False)
 
    elif args.lexicon == "data/lexicon_small/hate_lexicon_small.txt":

        features.to_csv(
            r'data/lexicon_small/hate_lexicon_small_df.csv', index=False)
        print("writing train data")
        train.to_csv(r'data/lexicon_small/train_features_lexicon_small.csv', index=False, header=False)
        print("Writing test data")
        test.to_csv(r'data/lexicon_small/test_features_lexicon_small.csv', index=False, header=False)
        print("writing dev data")
        dev.to_csv(r'data/lexicon_small/dev_features_lexicon_small.csv', index=False, header=False)
    
    print(df_transformed.shape)
    print(len(lexicon))
    print(f"Train set{train.shape}")
    print(f"Test set: {test.shape}")
    print(f"Dev set: {dev.shape}")

 

