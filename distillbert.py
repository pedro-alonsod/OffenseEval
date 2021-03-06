# -*- coding: utf-8 -*-
"""DistillBERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ekDnpL6nuAGwFxRmNO4iQceesrj1E5Yo
"""

# !pip install transformers
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import os
import sys
import tensorflow as tf
import gc
from finetune.base_models import BERT, BERTLarge, GPT2, GPT2Medium, GPT2Large, TextCNN, TCN, RoBERTa, DistilBERT, RoBERTa
# import finetue.regressor as Regressor
# from finetune import regressor as Regressor
print('inported finetune only regressor')
from finetune import *
print('imported all things')
# import labelencoder
from sklearn.preprocessing import LabelEncoder

import codecs
# instantiate labelencoder object
le = LabelEncoder()

np.random.seed(42)
print(os.getcwd())
print('Imported')

# torch.cuda.set_device(device)[SOURCE]

# Sets the current device.
# Usage of this function is discouraged in favor of device. In most cases it’s better to use CUDA_VISIBLE_DEVICES environmental variable.
# Parameters
# device (torch.device or python:int) – selected device. This function is a no-op if this argument is negative.
print(torch.cuda.device_count())
print(torch.cuda.get_device_capability(device=None))
device = torch.device("cuda: 3" if torch.cuda.is_available() else "cpu")
deviceData = torch.device("cuda: 2" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device=3))
# df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
# df = pd.read_csv('./OLIDv1.0/olid-training-v1.0.tsv', delimiter='\t', header='infer')#, nrows=3000) # Load smaller pieces nronws max atm rows working 1153
lstOfTweets = []
lstOfScores = []
# readFile = codecs.open('./OffenEval2020/OffenEval2020/task_A1_table_Pedro.txt', 'r', 'utf-8')
# readFile = codecs.open('./DataSets/HASOC/preprocessed/HASOC_Train_USER_URL_EmojiRemoved_Pedro.txt', 'r', 'utf-8')
hasonDf = pd.read_csv('./DataSets/HASOC/preprocessed/HASOC_Train_USER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
print(hasonDf.head(10))
sys.exit()
# i = 0
# part = 1
# task = 'A2'
# print('tasK:', task)
# next(readFile)
# linesToRead = 18
# chunksEnd = 40000
# chunksStart = 0
# for idx, line in enumerate(readFile):
#     # print(line)
#     tweetsTuple = line.split() 
#     tweet = ' '.join(tweetsTuple[:-1])
#     score = tweetsTuple[-1]  
#     # print('tweet'.upper(), tweet,'score'.upper(), score)
#     lstOfTweets.append(tweet)
#     lstOfScores.append(score)
#     i += 1
#     # if i > 850000:
#     #     break
# Model code start
def trainModelForBatches(tweets, scores, batchStart, batchEnd, task):
    # while part < linesToRead:
    #     if part*chunksEnd > i:
    #         batchTweets = lstOfTweets[chunksStart:]
    print('len of tweets to consider at the moment:', len(tweets), 'len of scores', len(scores))
    df = pd.DataFrame(list(zip(tweets, scores)))
    # print(i)
    # if df.is_empty():
    #     print('empty df!!')
    #     sys.exit()

    # sys.exit()
    # 2000 r = 0.27 ok-ram
    # 20000 r = 0.67 ok-ram
    # 30000 r = 0.66 ok-ram
    # 40000 r = 0.66 ok-ram
    # 50000 r = 0.67 ok ram
    # 60000 r =0.67 ok ram
    # 70000 r = 0.65 ok-ram
    # 80000 r = 0.67 ok-ram
    # 90000 r = ? not-ram
    # 100000 r = ? not-ram
    print(df.head())
    # print(df['subtask_a'])
    # sys.exit()
    print('********************************************************')
    # apply le on categorical feature columns
    # X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
    # X[categorical_cols].head(10)
    # print(df['subtask_a'].apply(lambda col: le.fit_transform(col)))
    # Not working

    # df['subtask_a_encoded'] = 0
    print(df.head())
    print(df.shape)



    # labelEncodedTaskA = []
    # labelEncodedTaskB = []
    # labelEncodedTaskC = []

    # for row in df.itertuples():
    # #    print(row[4])
    #     if row[3] == 'OFF':
    #         labelEncodedTaskA.append(1)
    #     else:
    #         labelEncodedTaskA.append(0)

    #     if row[4] == 'TIN':
    #         labelEncodedTaskB.append(1)
    #     else:
    #         labelEncodedTaskB.append(0)
    # df['LabelEncodedTaskA'] = labelEncodedTaskA
    # df['LabelEncodedTaskB'] = labelEncodedTaskB
    # print(df.tweet[:5])
    # df[['subtask_a', 'subtask_a_encoded']] = df[['subtask_a', 'subtask_a_encoded']].apply(lambda colA, colB: if colA == 'OFF': print('ofensive', colB))

    # df[['subtask_a', 'subtask_a_encoded']] = df[['subtask_a', 'subtask_a_encoded']].apply(lambda colA, colB: print(colA, colB))
    # id, tweet, sA, sB, sC, sAE
    #  1    2    3   4   5    6
    # for row in df.itertuples():
        # print(row[3])
        # if row[3] == 'OFf':'
        #    row[6] = 1'
        # else:
        #     row[6] = 0


    #sys.exit()
    # batch_1 = df[:2000]
    # configuration
# configuration = DistilBertConfig()

# # Initializing a model from the configuration
# model = DistilBertModel(configuration)

# # Accessing the model configuration
# configuration = model.config
# Configuration parameters
# vocab_size (int, optional, defaults to 30522) – Vocabulary size of the DistilBERT model. Defines the different tokens that can be represented by the inputs_ids passed to the forward method of BertModel.

# max_position_embeddings (int, optional, defaults to 512) – The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

# sinusoidal_pos_embds (boolean, optional, defaults to False) – Whether to use sinusoidal positional embeddings.

# n_layers (int, optional, defaults to 6) – Number of hidden layers in the Transformer encoder.

# n_heads (int, optional, defaults to 12) – Number of attention heads for each attention layer in the Transformer encoder.

# dim (int, optional, defaults to 768) – Dimensionality of the encoder layers and the pooler layer.

# hidden_dim (int, optional, defaults to 3072) – The size of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.

# dropout (float, optional, defaults to 0.1) – The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.

# attention_dropout (float, optional, defaults to 0.1) – The dropout ratio for the attention probabilities.

# activation (str or function, optional, defaults to “gelu”) – The non-linear activation function (function or string) in the encoder and pooler. If string, “gelu”, “relu”, “swish” and “gelu_new” are supported.

# initializer_range (float, optional, defaults to 0.02) – The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

# qa_dropout (float, optional, defaults to 0.1) – The dropout probabilities used in the question answering model DistilBertForQuestionAnswering.

# seq_classif_dropout (float, optional, defaults to 0.2) – The dropout probabilities used in the sequence classification model DistilBertForSequenceClassification.
    config = ppb.RobertaConfig()
    # Initializing a DistilBERT 
    model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-base-uncased')

    ## Want BERT instead of distilBERT? Uncomment the following line:
    #model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = Classifier(base_model=RoBERTa)
    # model = Classifier()               # Load base model
    # model.fit(trainX, trainY)          # Finetune base model on custom data
    # predictions= model.predict(testX)  # ['class_2', 'class_1', 'class_3'...]
    # probs = model.predict_proba(testX) # [{'class_1': 0.23, 'class_2': 0.54, ..}, ..]
    # model.save(path)                   # Serialize the model to disk
    # model = model_class.from_pretrained(pretrained_weights)
    # model = torch.nn.parallel.DistributedDataParallel(model)
    #model.to(device)

    tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # tokenized = df.tweet.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    print(tokenized)
    #sys.exit()

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    print(padded)

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(np.array(padded))
    attention_mask = torch.tensor(attention_mask)

    #input_ids = input_ids.to(device)
    #attention_mask = attention_mask.to(device)

    print(attention_mask.shape)
    print('att Mask', attention_mask)

    print("\n * input_ids_tensor \n ")
    print(input_ids)
    print(input_ids.device)

    # print("\n * segment_ids_tensor \n ")
    # print(segment_ids_tensor)
    # print(segment_ids_tensor.device)

    print("\n * input_mask_tensor \n ")
    print(attention_mask)
    print(attention_mask.device)

    print("\n * self.device \n ")
    print(device)

    # sys.exit()

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    # Slice the output for the first position for all the sequences, take all hidden unit outputs
    # All the [CLS] tokens for classification purposes
    # features = last_hidden_states[0][:,0,:].numpy()

    # TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    features = last_hidden_states[0][:,0,:].cpu().numpy()
    dfFeaturesSave = pd.DataFrame(zip(features, df[1]))
    dfFeaturesSave.to_csv(f'./FeaturesCompilation/FeaturesAndLabelsForTask{task}BatchStart{batchStart}BatchEnd{batchEnd}.csv')
    # labels = df['LabelEncodedTaskA']
    # labels = batch_1[1]
    # labels = df[1]
    # # del df
    del dfFeaturesSave
    # del model
    del tokenizer
    del tokenized
    del padded
    del attention_mask
    del input_ids
    del last_hidden_states
    gc.collect()


    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    # # dfFeaturesSave = pd.DataFrame()
    # # Save to file instead of tweet the features like the input but for the features id, features, scores

    # # lr_clf = LogisticRegression()
    # # lr_clf.fit(train_features, train_labels)
    # # # Now that the model is trained, we can score it against the test set:

    # # print(lr_clf.score(test_features, test_labels))
    # print(i)
    # print
    # lr = LinearRegression()
    # lr.fit(train_features, train_labels)

    # print(lr.score(test_features, test_labels))
# Model code END ****************************************************************************************

print(i)
print(len(lstOfTweets), lstOfTweets[0])
startBatch = 0
rangeStart = 40000
step = 40000
for i in range(rangeStart, len(lstOfTweets), step):
    # startBatch = 0
    print('range of batches', startBatch, i)
    # print('until idx', i)
    if i  >= 1040000:
        # print('len of last bathc', len(lstOfTweets[i:]))
        trainModelForBatches(lstOfTweets[i:], lstOfScores[i:], i, ':', task)
        break
    # print('len of normal batches', len(lstOfTweets[startBatch:i]))#, 'first tweet of batch.---*', lstOfTweets[startBatch])
    trainModelForBatches(lstOfTweets[startBatch:i], lstOfScores[startBatch:i], startBatch, i, task)
    startBatch += 40000

print('complete!!')
# RoBERTa has provided state-of-the-art results on a variety of natural language tasks, as of late 2019
# model = Classifier(base_model=DistilBERT)

# The GPT and GPT2 model families allow experimentation with text generation
# model = LanguageModel(base_model=GPT2)

# DistilBERT offers competetive finetuning performance with faster training and inference times thanks to its low parameter count
# model = Classifier(base_model=DistilBERT)

sys.exit()
