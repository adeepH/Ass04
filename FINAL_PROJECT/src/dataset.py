import torch
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertForSequenceClassification, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification, XLMRobertaTokenizer, AutoTokenizer,AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, label):
        self.encodings = encodings
        self.label = label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)


def preprocess(train_path, dev_path):
# Load and preprocess dataset
# Assumes the dataset is already preprocessed and split into train and test sets
#train_dataset = ...  # Load and preprocess training data
#test_dataset = ...   # Load and preprocess test data 

  train = pd.read_csv(train_path, index_col=0)
  train['label'] = train['task1_label'].apply({'YES':1, 'NO':0}.get)

  train_dataset = train.drop(columns=['language', 'gender', 'task1_label'])
  train_dataset = train_dataset.reset_index()
  dev = pd.read_csv(dev_path, index_col=0)
  dev['label'] = dev['task1_label'].apply({'YES':1, 'NO':1}.get)
  test_dataset = dev.drop(columns=['language', 'gender', 'task1_label'])
  test_dataset = test_dataset.reset_index()

  return train_dataset, test_dataset