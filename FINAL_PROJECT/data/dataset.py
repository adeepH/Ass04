import pandas as pd
import numpy as np

df = pd.read_json('training/EXIST2023_training.json') 
df = pd.read_json('dev/EXIST2023_dev.json')

class generate_labels():

    def __init__(self, df):

        self.df = df.T
        self.task1_outputs = self.df.labels_task1
        self.task1_label = []
        self.tweet = self.df.tweet
        self.lang = self.df.lang
        self.gender = self.df.gender_annotators

        """ 
    def export_df(self):

        return pd.DataFame({
            'tweet':self.tweet,
            'task1_label': self.task1_label,
            'gender': self.gender,
            'language': self.lang
        }) 
        """

    def majority_voting(self):
        
        for task1_output in self.task1_outputs:

            majority_label = max(set(task1_output), key=task1_output.count)
            self.task1_label.append(majority_label)

        return pd.DataFrame({
            'tweet':self.tweet,
            'task1_label': self.task1_label,
            'gender': self.gender,
            'language': self.lang
        }) 
 

label_generator = generate_labels(df)

majority_df = label_generator.majority_voting()
#weighted_df = label_generator.weighted_voting()
print(majority_df.head(15))

majority_df.to_csv('dev.csv', index=0)
#print(weighted_df.head(15))