import torch
from torch.utils.data import DataLoader
from dataset import CustomDataSet
from transformers import BertTokenizer, BertForSequenceClassification, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification, XLMRobertaTokenizer, AutoTokenizer,AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

train_path = 'train.csv'
dev_path = 'dev.csv'
train_dataset, test_dataset = preprocess(train_path, dev_path)
    
model_dict = {
    'xlmr': 'xlm-roberta-base',
    'mbert': 'bert-base-multilingual-cased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base'
}
for model in model_dict.keys():
  # Free up CUDA memory
  torch.cuda.empty_cache()
  print(f'Using pretrained Language Model {model}')
  #tokenizer = tokenizer_dict[model]

  # load the tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_dict[model])

  # tokenize the text
  train_encodings = tokenizer(train_dataset['tweet'].tolist(), truncation=True, padding=True)
  test_encodings = tokenizer(test_dataset['tweet'].tolist(), truncation=True, padding=True)

  # Conver tokenized data into torch tensors
  # create a dataset object to feed into data loaders
  train_dataset = CustomDataset(train_encodings, train_dataset['label'].tolist())
  test_dataset = CustomDataset(test_encodings, test_dataset['label'].tolist())

  # create the data loaders for the custom datasets
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  # Load the large language model 
  #model = model_dict[model]
  model = AutoModelForSequenceClassification.from_pretrained(model_dict[model], num_labels=2)
  
  # Use cuda if available
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Save the model in the GPU
  model.to(device)

  # Setting up the optimzers and other parameters
  optimizer = AdamW(model.parameters(), lr=2e-5) # can change it later if necessary

  # Number of epochs
  NUM_EPOCHS = 6

  # Set up the training steps and scheduler
  num_training_steps = len(train_loader) * NUM_EPOCHS
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

  # Fine-tuning loop
  for epoch in range(NUM_EPOCHS):

    # set the model to train
    model.train()

    # Batch Gradient descent and optimizers
    for batch in train_loader:
      
      # Initialize zero gradients
      optimizer.zero_grad()
      
      # Feed input data to models both on same device (GPU or CPU)
      inputs = {key:val.to(device) for key, val in batch.items() if key != 'label'}
      labels = batch['label'].to(device)

      # Get outputs from model
      outputs = model(**inputs, labels=labels)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      scheduler.step()

    # Evaluating on the dev set after each epoch
    model.eval()
    
    # Define true and preds as list
    y_true = []
    y_pred = []

    with torch.no_grad():
      for batch in test_loader:
        inputs = {key:val.to(device) for key, val in batch.items() if key!= 'label'}
        labels = batch['label'].to(device)
        outputs = model(**inputs, labels=labels)

        # get logits from models
        logits = model.logits
        preds = torch.argmax(logits, dim=1)
        y_true.extend(batch['label'].tolist())
        y_pred.extend(preds.tolist())

      # Evaluation on the prediction
      accuracy = accuracy_score(y_true, y_pred)
      f1 = f1_score(y_true, y_pred)
      print(f'Epoch {epoch + 1}: Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}')
  
