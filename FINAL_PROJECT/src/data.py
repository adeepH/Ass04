import pandas as pd

train = pd.read_json('CSCI-B-659/FINAL_PROJECT/training/EXIST2023_training.json') 
dev = pd.read_json('CSCI-B-659/FINAL_PROJECT/dev/EXIST2023_dev.json')

class generate_labels():
    def __init__(self, df):

        self.df = df.T
        self.task1_outputs = self.df.labels_task1
        self.task1_label = []
        self.tweet = self.df.tweet
        self.lang = self.df.lang
        self.gender = self.df.gender_annotators
 
    def majority_voting(self):
 
        for task1_output in self.task1_outputs:
            majority_label = max(set(task1_output), key=task1_output.count)
            self.task1_label.append(majority_label)

        return pd.DataFrame({
            'text':self.tweet,
            'label': self.task1_label,
         #   'gender': self.gender,
         #   'language': self.lang
        }) 
label_generator = generate_labels(train)
majority_df = label_generator.majority_voting()
#weighted_df = label_generator.weighted_voting()
print(majority_df.head(15))
majority_df['label'] = majority_df['label'].apply({'YES':1, 'NO':0}.get})
majority_df.to_csv('train.csv', index=0)

label_generator = generate_labels(dev)
dev_df = label_generator.majority_voting()
dev_df['label'] = dev_df['label'].apply({'YES':1, 'NO':0}.get})
dev_df.to_csv('dev.csv', index=0)
