#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('nvidia-smi')


import pandas as pd
import numpy as np



imdb_df = pd.read_csv('IMDB_Dataset.csv', index_col = None)

print(len(imdb_df))

#imdb_df.head(3)

#label = []
#
#for index, row in imdb_df.iterrows():
#    if(row['sentiment'] == 'positive'): label.append(1)
#    else: label.append(0)
#
#
#imdb_df['label'] = label
#
#imdb_df = imdb_df.drop(['sentiment'], axis = 1)
#
#
#
#imdb_df = imdb_df.head(10000)
#
#
## In[ ]:
#
#
#imdb_df.label.value_counts()
#
#
## In[ ]:
#
#
#PRETRAINED_MODEL_NAME = 'roberta-large'
#PRETRAINED_MODEL_PATH = '../models/' + PRETRAINED_MODEL_NAME
#
#
## In[ ]:
#
#
#from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
#
#
## In[ ]:
#
#
#roberta_model = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH)
#roberta_tok = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
#
#
## In[ ]:
#
#
#import torch
#from sklearn.model_selection import train_test_split
#
#
## In[ ]:
#
#
#class CreateDataset(torch.utils.data.Dataset):
#    def __init__(self, reviews, labels, tokenizer, max_len):
#        self.reviews = reviews
#        self.labels = labels
#        self.tokenizer = tokenizer
#        self.max_len = max_len
#        
#    def __len__(self):
#        return len(self.reviews)
#    
#    def __getitem__(self, item):
#        review = str(self.reviews[item])
#        label = self.labels[item]
#        
#        encoding = self.tokenizer.encode_plus(review, 
#                                             add_special_tokens = True,
#                                             max_length = self.max_len, 
#                                             truncation = True,
#                                             return_tensors = 'pt',
#                                             return_token_type_ids = False,
#                                             return_attention_mask = True,
#                                             padding = 'max_length')
#        
#        return{
#            'review_text': review,
#            'input_ids' : encoding['input_ids'].flatten(),
#            'attention_mask' : encoding['attention_mask'].flatten(),
#            'labels' : torch.tensor(label, dtype=torch.long)            
#        }
#
#
## In[ ]:
#
#
#df_train, df_val = train_test_split(imdb_df, test_size = 0.3, random_state = 2021)
#print(df_train.shape, df_val.shape)
#
#
## In[ ]:
#
#
#print(df_train.label.value_counts())
#print(df_val.label.value_counts())
#
#
## In[ ]:
#
#
#def create_data_loader(df, tokenizer, max_len, batch_size):
#    ds = CreateDataset(reviews = df.review.to_numpy(),
#                       labels = df.label.to_numpy(),
#                       tokenizer = tokenizer,
#                       max_len = max_len
#                      )
#    
#    return torch.utils.data.DataLoader(ds, 
#                                       batch_size = batch_size, 
#                                       num_workers = 4)
#
#
## In[ ]:
#
#
#MAX_LEN = 512
#BATCH_SIZE = 8
#
#train_data_loader = create_data_loader(df_train, roberta_tok, MAX_LEN, BATCH_SIZE)
#val_data_loader = create_data_loader(df_val, roberta_tok, MAX_LEN, BATCH_SIZE)
#
#
## In[ ]:
#
#
#check_data = next(iter(train_data_loader))
#check_data.keys()
#
#
## In[ ]:
#
#
## Uncomment and run this cell to visualize the roberta-large architecture
##roberta_model
#
#
## In[ ]:
#
#
## Embedding layer
##roberta_model.roberta.embeddings
#
#
## In[ ]:
#
#
## Encoder Layers
##roberta_model.roberta.encoder
#
#
## In[ ]:
#
#
## Classifier Layer
##roberta_model.classifier
#
#
## In[ ]:
#
#
#class MultiGPUClassifier(torch.nn.Module):
#    def __init__(self, roberta_model):
#        super(MultiGPUClassifier, self).__init__()
#        self.embedding = roberta_model.roberta.embeddings.to('cuda:0')
#        self.encoder = roberta_model.roberta.encoder.to('cuda:1')
#        self.classifier = roberta_model.classifier.to('cuda:1')
#        
#    def forward(self, input_ids, token_type_ids = None, attention_mask = None, labels = None):
#        emb_out = self.embedding(input_ids.to('cuda:0'))
#        enc_out = self.encoder(emb_out.to('cuda:1'))
#        classifier_out = self.classifier(enc_out[0])
#        return classifier_out        
#
#
## In[ ]:
#
#
#get_ipython().system('nvidia-smi')
#
#
## In[ ]:
#
#
#multi_gpu_roberta = MultiGPUClassifier(roberta_model)
#
#
## In[ ]:
#
#
#from transformers import get_linear_schedule_with_warmup, AdamW
#
#
## In[ ]:
#
#
#EPOCHS = 2
#LR = 1e-5
#
#optimizer = AdamW(multi_gpu_roberta.parameters(), lr = LR)
#total_steps = len(train_data_loader) * EPOCHS
#
#scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                           num_warmup_steps = 0, 
#                                           num_training_steps = total_steps)
#
#
## In[ ]:
#
#
#loss_fn = torch.nn.CrossEntropyLoss().to('cuda:1')
#
#
## In[ ]:
#
#
#def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
#    model = model.train()
#    losses = []
#    correct_predictions = 0
#    
#    for d in data_loader:
#        input_ids = d['input_ids']
#        attention_mask = d['attention_mask']
#        reshaped_attention_mask = attention_mask.reshape(d['attention_mask'].shape[0], 1, 1, d['attention_mask'].shape[1])
#        targets = d['labels']
#        
#        outputs= model(input_ids = input_ids, attention_mask = reshaped_attention_mask)
#        _, preds = torch.max(outputs, dim = 1)
#        loss = loss_fn(outputs, targets.to('cuda:1'))
#        
#        correct_predictions += torch.sum(preds == targets.to('cuda:1'))
#        losses.append(loss.item())
#        
#        loss.backward()
#        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
#        optimizer.step()
#        scheduler.step()
#        optimizer.zero_grad()
#        
#    return correct_predictions.double() / n_examples, np.mean(losses)
#        
#
#
## In[ ]:
#
#
#def eval_model(model, data_loader, loss_fn, n_examples):
#    model = model.eval()
#    losses = []
#    correct_predictions = 0
#    
#    with torch.no_grad():
#        for d in data_loader:
#            input_ids = d['input_ids']
#            attention_mask = d['attention_mask']
#            reshaped_attention_mask = attention_mask.reshape(d['attention_mask'].shape[0], 1, 1, d['attention_mask'].shape[1])
#            targets = d['labels']
#            
#            outputs = model(input_ids = input_ids, attention_mask = reshaped_attention_mask)
#            _, preds = torch.max(outputs, dim = 1)
#            
#            loss = loss_fn(outputs, targets.to('cuda:1'))
#            
#            correct_predictions += torch.sum(preds == targets.to('cuda:1'))
#            losses.append(loss.item())
#            
#        return correct_predictions.double() / n_examples, np.mean(losses)
#
#
## In[ ]:
#
#
#from collections import defaultdict
#
#history = defaultdict(list)
#best_accuracy = 0
#
#
## In[ ]:
#
#
#get_ipython().run_cell_magic('time', '', "\nfor epoch in range(EPOCHS):\n    print(f'Epoch {epoch + 1}/{EPOCHS}')\n    print('-' * 10)\n    \n    train_acc, train_loss = train_model(multi_gpu_roberta, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))\n    print(f'Train Loss: {train_loss} ; Train Accuracy: {train_acc}')\n    \n    val_acc, val_loss = eval_model(multi_gpu_roberta, val_data_loader, loss_fn, len(df_val))\n    print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')\n    \n    print()\n    \n    history['train_acc'].append(train_acc)\n    history['train_loss'].append(train_loss)\n    history['val_acc'].append(val_acc)\n    history['val_loss'].append(val_loss)\n    \n    if val_acc > best_accuracy:\n        torch.save(multi_gpu_roberta.state_dict(), 'multi_gpu_roberta_best_model_state.bin')\n        best_acc = val_acc\n")
#
#
## In[ ]:
#
#
#import matplotlib.pyplot as plt
#
#
## In[ ]:
#
#
#plt.plot(history['train_acc'], label='train accuracy')
#plt.plot(history['val_acc'], label='validation accuracy')
#plt.title('Training history')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend()
#plt.ylim([0, 1]);
#
#
## In[ ]:
#
#
#def get_predictions(model, data_loader):
#    model = model.eval()
#    review_texts = []
#    predictions = []
#    prediction_probs = []
#    real_values = []
#    with torch.no_grad():
#        
#        for d in data_loader:
#
#            texts = d["review_text"]
#            input_ids = d["input_ids"]
#            attention_mask = d["attention_mask"]
#            labels = d["labels"]
#
#            outputs = model(input_ids=input_ids,
#                            attention_mask=attention_mask)
#
#            _, preds = torch.max(outputs, dim=1)
#            review_texts.extend(texts)
#            predictions.extend(preds)
#            prediction_probs.extend(outputs)
#            real_values.extend(labels)
#        predictions = torch.stack(predictions).cpu()
#        prediction_probs = torch.stack(prediction_probs).cpu()
#        real_values = torch.stack(real_values).cpu()
#    return review_texts, predictions, prediction_probs, real_values
#
#
## In[ ]:
#
#
#y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(multi_gpu_roberta, val_data_loader)
#
#
## In[ ]:
#
#
#from sklearn.metrics import classification_report, confusion_matrix
#import seaborn as sns
#
#
## In[ ]:
#
#
#class_names = ['negative', 'positive']
#print(classification_report(y_test, y_pred, target_names=class_names))
#
#
## In[ ]:
#
#
#def show_confusion_matrix(confusion_matrix):
#    
#    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
#    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
#    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
#    plt.ylabel('True sentiment')
#    plt.xlabel('Predicted sentiment');
#    
#cm = confusion_matrix(y_test, y_pred)
#df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
#show_confusion_matrix(df_cm)
#
#
## In[ ]:
#
#
#idx = 1
#review_text = y_review_texts[idx]
#true_sentiment = y_test[idx]
#
#pred_df = pd.DataFrame({'class_names': class_names, 
#                        'values': y_pred_probs[idx] 
#                       })
#
#print(review_text)
#print()
#print(f'True sentiment: {class_names[true_sentiment]}')
#
#
## In[ ]:
#
#
#sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
#plt.ylabel('sentiment')
#plt.xlabel('probability')
#plt.xlim([0, 1]);
#
