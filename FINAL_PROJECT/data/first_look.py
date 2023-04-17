import pandas as pd
import numpy as np

#path = 'training'
df = pd.read_csv('majority_voting.csv')
print(df.columns)
df['task1_label'] = df['task1_label'].apply({'YES': 1, 'NO': 0}.get)
print(df['task1_label'].value_counts())
df2 = pd.DataFrame({
    'text' : df['tweet'],
    'label': df['task1_label'],
    'lang': df['language']
})
df2.to_csv('binary_labels.csv', index=False)


                                                                        
                                                                