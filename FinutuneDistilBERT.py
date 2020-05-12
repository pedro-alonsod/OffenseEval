from simpletransformers.classification import ClassificationModel
# from simpletransformers.regression import RegressionModel
import simpletransformers
import pandas as pd
import codecs
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import os

lstOfTweets = []
lstOfScores = []
# print(simpletransformers.__version__)
# sys.exit()
readFile = codecs.open('./OffenEval2020/OffenEval2020/task_A1_table_Pedro.txt', 'r', 'utf-8')
# i = 0
part = 1
task = 'A1-A6'
print('tasK:', task)
next(readFile)
linesToRead = 18
chunksEnd = 40000
chunksStart = 0
train = True
if train:
    for tweetsFile in ['./OffenEval2020/OffenEval2020/task_A1_table_Pedro.txt']:
    # for tweetsFile in ['./OffenEval2020/OffenEval2020/task_A1_table_Pedro.txt', './OffenEval2020/OffenEval2020/task_A2_table_Pedro.txt', './OffenEval2020/OffenEval2020/task_A3_table_Pedro.txt', 
    #                 './OffenEval2020/OffenEval2020/task_A4_table_Pedro.txt', './OffenEval2020/OffenEval2020/task_A5_table_Pedro.txt', './OffenEval2020/OffenEval2020/task_A6_table_Pedro.txt']:
        readFile = codecs.open(tweetsFile, 'r', 'utf-8')
        next(readFile)
        i = 0
        for idx, line in enumerate(readFile):
            # print(line)
            tweetsTuple = line.split() 
            tweet = ' '.join(tweetsTuple[:-1])
            score = float(tweetsTuple[-1])  
            # print('tweet'.upper(), tweet,'score'.upper(), score)
            lstOfTweets.append(tweet)
            lstOfScores.append(score)
            i += 1
            # if i > 850000:
            #     break
    print('total size of tweets', i)
    print('lens', len(lstOfTweets), len(lstOfScores))
# lsys.exit()
olidTrainDf = pd.read_csv('DataSets/olid-training-v1.0.tsv', delimiter='\t')
print(olidTrainDf.head())
# sys.exit()

# labelsSubtaskA = []
# labelsSubtaskB = []
# labelsSubtaskC = []
# for row in olidTrainDf.itertuples():
#     # print(row)
#     # task a
#     if row[3] == 'OFF':
#         labelsSubtaskA.append(int(1))
#         # print('is OFF')
#     elif row[3] == 'NOT':
#         labelsSubtaskA.append(int(0))
#         # print('is NOT')
#     # task b
#     if row[4] == 'UNT':
#         labelsSubtaskB.append(int(0))
#     elif row[4] == 'TIN':
#         labelsSubtaskB.append(int(1))
#     elif row[4] != 'UNT' or row[4] != 'TIN':
#         # print('is NaN here!!!')
#         labelsSubtaskB.append(np.nan)

#     # task c
#     if row[5] == 'IND':
#         labelsSubtaskC.append(int(0))
#     elif row[5] == 'GRP':
#         labelsSubtaskC.append(int(1))
#     elif row[5] != 'IND' or row[5] != 'GRP':
#         # print('is na hereee!!!')
#         labelsSubtaskC.append(np.nan)

# print('len', len(labelsSubtaskA), len(labelsSubtaskB), len(labelsSubtaskC), labelsSubtaskC)


# print(olidTrainDf.isna().any())
# olidTrainDf['LabelEncodedA'] = labelsSubtaskA
# olidTrainDf['LabelEncodedB'] = labelsSubtaskB
# olidTrainDf['LabelEncodedC'] = labelsSubtaskC
# print(olidTrainDf.head(10))

# print(labelsSubtaskA)
# sys.exit()
if train:
    train_data = [
        ['Example sentence belonging to class 1', 'Yep, this is 1', 1.8],
        ['Example sentence belonging to class 0', 'Yep, this is 0', 0.2],
        ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 4.5]
    ]
    print('loading data train')
    train_df = pd.DataFrame(list(zip(lstOfTweets[:-3000], lstOfScores[:-3000])), columns=['tweets', 'scores'])
    # train_df = pd.DataFrame(olidTrainDf[['tweet', 'LabelEncodedA']].iloc[:-384])
    print(train_df.head(), train_df.shape)
    # print('last 3',train_df.iloc[-3:])
    # sys.exit()

    print('train loaded dropping NAs')
    train_df = train_df.dropna()#train_df.fillna(train_df.mean())
    print('NAs dropped train data')
    # train_df["scores"] = train_df["scores"].apply(lambda x: list(map(float, x)))
    print('head of :-1000 df', train_df.head(-5))
    # print('describe', train_df.describe())
    # print('train nan', train_df.isna().any())

    # train_df = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'labels'])

    eval_data = [
        ['Example sentence belonging to class 1', 'Yep, this is 1', 1.9],
        ['Example sentence belonging to class 0', 'Yep, this is 0', 0.1],
        ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 5]
    ]

    # eval_df = pd.DataFrame(eval_data, columns=['text_a', 'text_b', 'labels'])
    print('loading eval data')
    eval_df = pd.DataFrame(list(zip(lstOfTweets[-3000:], lstOfScores[-3000:])), columns=['tweets', 'scores'])
    # eval_df = train_df.dropna()[-3000:]#train_df.fillna(train_df.mean())
    print('NAs dropped eval data')
    # eval_df = pd.DataFrame(olidTrainDf[['tweet', 'LabelEncodedA']].iloc[-384:])
    print('eval data loaded', eval_df.shape)
    # eval_df["scores"] = eval_df["scores"].apply(lambda x: list(map(float, x)))
    print('head of -1000: eval', eval_df.head())
    # print('describe', eval_df.describe())
    # print('eval nan', eval_df.isna().any())
    # sys.exit()
    train_df_chunks = np.array_split(train_df, 5)
    print(train_df_chunks[0][:10])
    for i in range(5):
        print(train_df_chunks[i].shape)

def trainModel():

    train_args= {
    "output_dir": "outputsRoberta3Full", # save things
    "cache_dir": "cache", #cache dir
    "save_model_every_epoch": True, #every eåoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": False,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 3,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 3, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': True,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 2000,
    "evaluate_during_training_verbose": True,
    }

    print('Create a ClassificationModel/regression')

    # model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=1, use_cuda=True, cuda_device=0, args=train_args) #first
    model = ClassificationModel('roberta', 'roberta-base', num_labels=1, use_cuda=True, cuda_device=0, args=train_args)
    # print(train_df.head())

    print('Train the model')
    model.train_model(train_df, eval_df=eval_df)

    # Evaluate the model
    # result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    # print('result', result)
    # print('wrong_prediction', wrong_predictions)
    # predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
    # print('predictions...')
    # print(predictions)
    # print(raw_outputs)

def loadModelSaved():
    train_args= {
    "output_dir": "outputs/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every eåoch save it
    "save_eval_checkpoints": False,
    "use_early_stopping": False,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 30,
    # "num_train_epochs": 50,
    "train_batch_size": 64,
    "n_gpu": 2, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': True,
    }
    # model = ClassificationModel('distilbert', 'outputs', num_labels=1, use_cuda=True, cuda_device=0, args=train_args)
    model = ClassificationModel('roberta', 'outputsRoberta3Full', num_labels=1, use_cuda=True, cuda_device=2, args=train_args)# outputsModelRoberta previous
    # result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    # print('result', result)
    # print('wrong_prediction', wrong_predictions)


def loadOlidOffenseEvalData():
    # olidTestDf = pd.read_csv('DataSets/NewOLIDfiles/OLID_TEST_A_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')#, nrows=3310)
    # olidLabelsA = pd.read_csv('DataSets/NewOLIDfiles/labels-levela.csv', delimiter=',')
    olidTrainDf = pd.read_csv('DataSets/NewOLIDfiles/OLID_Tain_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')#, nrows=3310)
    # offenseTestDF = pd.read_csv('DataSets/OffenseEvalTest/Offenseval_Test_Set_Preprocessed.txt', delimiter='\t')
    # offenseLabelA = pd.read_csv('DataSets/OffenseEvalTest/englishA-goldlabels.csv', delimiter=',', header=None)
    # print(offenseTestDF.head())
    # print(offenseTestDF.shape)
    # print(offenseLabelA.head())
    # print(offenseLabelA.shape)
    # print(olidTestDf.head())
    # print(olidTestDf.shape)
    # print(olidLabelsA.head())
    # print(olidLabelsA.shape)
    print(olidTrainDf.head())
    print(olidTrainDf.shape)
    # sys.exit()
    # offenseTestDF['Labels'] = offenseLabelA[1]
    # print(offenseTestDF.head(10))
    # olidTestDf['Labels'] = olidLabelsA[1] #not for olid
    # olidTestDf['Labels'] = olidLabelsA['label']
    # print(olidTestDf.head(10))
    print(olidTrainDf.head(10))
    # sys.exit()
#    # olidTrainDf = pd.read_csv('DataSets/Olid/OlidTraining_Pedro.txt', delimiter='\t')#, nrows=3310)
    # print(olidTrainDf.shape)
    # print(olidTrainDf.head())


    # labelsSubtaskA = []
    # labelsSubtaskB = []
    # labelsSubtaskC = []
    # for row in olidTrainDf.itertuples():
    #     # print(row)
    #     # task a
    #     if row[3] == 'OFF':
    #         labelsSubtaskA.append(int(1))
    #         # print('is OFF')
    #     elif row[3] == 'NOT':
    #         labelsSubtaskA.append(int(0))
    #         # print('is NOT')
    #     # task b
    #     if row[4] == 'UNT':
    #         labelsSubtaskB.append(int(0))
    #     elif row[4] == 'TIN':
    #         labelsSubtaskB.append(int(1))
    #     elif row[4] != 'UNT' or row[4] != 'TIN':
    #         # print('is NaN here!!!')
    #         labelsSubtaskB.append(np.nan)

    #     # task c
    #     if row[5] == 'IND':
    #         labelsSubtaskC.append(int(0))
    #     elif row[5] == 'GRP':
    #         labelsSubtaskC.append(int(1))
    #     elif row[5] != 'IND' or row[5] != 'GRP':
    #         # print('is na hereee!!!')
    #         labelsSubtaskC.append(np.nan)

    # print('len', len(labelsSubtaskA), len(labelsSubtaskB), len(labelsSubtaskC), labelsSubtaskC)


    # print(olidTrainDf.isna().any())
    # olidTrainDf['LabelEncodedA'] = labelsSubtaskA
    # olidTrainDf['LabelEncodedB'] = labelsSubtaskB
    # olidTrainDf['LabelEncodedC'] = labelsSubtaskC
    # print(olidTrainDf.head(10))

    # print(labelsSubtaskA)
    # sys.exit()
    # print(olidTestDf['tweet'])
    # prections(olidTestDf)
    prections(olidTrainDf)
    # prections(offenseTestDF)

def prections(dataframe):
    train_args= {
    "output_dir": "outputs/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": False, #every eåoch save it
    'reprocess_input_data': False,
    'overwrite_output_dir': False,
    'num_train_epochs': 3,
    # "num_train_epochs": 50,
    "train_batch_size": 64,
    "n_gpu": 2, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': True,
    }
    # model = ClassificationModel('distilbert', 'outputs2', num_labels=1, use_cuda=True, cuda_device=2, args=train_args)
    model = ClassificationModel('roberta', 'outputsRoberta3Full', num_labels=1, use_cuda=True, cuda_device=3, args=train_args)
    sentencesForPrediction = []
    compareLabels = []
    for row in dataframe.itertuples():
        # sentencesForPrediction.append([row[2]])
        # predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        print('row[0]', row[0],'row[1]', row[1], 'r3', row[3], 'r2', row[2])
        predictions, raw_outputs = model.predict([row[2]])#[["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        # print('pred', prections, predictions > 0.1)
        print('raw', raw_outputs)
        sentencesForPrediction.append((row[1], raw_outputs, row[3]))
        # print(sentencesForPrediction)
        # sys.exit()
    print('results', sentencesForPrediction)
    # predictions, raw_outputs = model.predict(sentencesForPrediction)#[["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
    # print('len', len(raw_outputs), raw_outputs)
    # for elem in sentencesForPrediction:
    #     print(elem)
    #     compareLabels.append(elem
    dataframe['LabelsTaskACompare'] = sentencesForPrediction
    outputToPandas = pd.DataFrame(sentencesForPrediction)
    print(outputToPandas.head())
    outputToPandas.to_csv('./compareOut/RobertaOlidTrain1epoch.csv', sep='\t')
    print(dataframe.head(50))

def findCutoff():
    df = pd.read_csv('./compareOut/outputs.csv', delimiter='\t')
    print(df.head(10))
    cutOff = 0.5
    testAcc = 0.0
    lstOfPred = []
    for elem in df.itertuples():
        print(elem[3])
        if elem[3] > cutOff:
            lstOfPred.append('OFF')
        else:
            lstOfPred.append('NOT')
    # print(lstOfPred)
    print('truth list', df['0'].tolist())
    print('predicted list', lstOfPred)
    # print('equality', (df['0'].tolist() == lstOfPred))
    for x, y in zip(df['0'].tolist(), lstOfPred):
        if x == y:
            testAcc += 1

    print('acc mine', testAcc/len(lstOfPred)) #first 10 = 0.9
    result = accuracy_score(df['0'], lstOfPred)
    print(result)

    macro = f1_score(df['0'], lstOfPred, average='macro')
    print(macro)

def evalTestSet():
    dataframe = pd.read_csv('DataSets/public_data_A/test_a_tweets.tsv', delimiter='\t')#, nrows=5)
    # print(dataframe.head())
    # print(dataframe.shape)
    # print(dataframe['tweet'])
    # prections(dataframe)
    train_args= {
    "output_dir": "outputs/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": False, #every eåoch save it
    'reprocess_input_data': False,
    'overwrite_output_dir': False,
    'num_train_epochs': 3,
    # "num_train_epochs": 50,
    "train_batch_size": 64,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': True,
    }
    # model = ClassificationModel('distilbert', 'outputsDistilBERT20', num_labels=1, use_cuda=True, cuda_device=2, args=train_args)
    model = ClassificationModel('roberta', 'outputsModelRoberta', num_labels=1, use_cuda=True, cuda_device=3, args=train_args)
    sentencesForPrediction = []
    lstOfIdx = []
    compareLabels = []
    for row in dataframe.itertuples():
        # sentencesForPrediction.append([row[2]])
        # predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        # print(row)
        # print('row[0]', row[0],'row[1]', row[1], 'row[2]', row[2])
        # sys.exit()
        predictions, raw_outputs = model.predict([row[2]])#[["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        # print('pred', prections, predictions > 0.1)
        # print('raw', raw_outputs)
        print('id', row[1], 'tweet', row[2], ' score > .42 OFF', raw_outputs)
        lblClass = ''
        if raw_outputs > 0.42: #eyeballing 70 seem to high 0.60 too low
            lblClass = 'OFF'
        else:
            lblClass = 'NOT'
        sentencesForPrediction.append((row[1], lblClass))
        print('id', row[1], 'tweet', row[2], ' score > .42 OFF', raw_outputs, 'label', lblClass)
        # print(sentencesForPrediction)
    # sys.exit()
    print('results', sentencesForPrediction)
    # dataframe['LabelsTaskACompare'] = sentencesForPrediction
    outputToPandas = pd.DataFrame(sentencesForPrediction)
    print(outputToPandas.head())
    outputToPandas.to_csv('./EvalREsults/ResultsRoberta.csv', sep=',', index=False, header=False)
    print(dataframe.head(50))


def retrainModel():
    train_args= {
    "output_dir": "outputsClass/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 3, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': True,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }

    model = ClassificationModel('roberta', 'outputsModelRoberta', num_labels=1, use_cuda=True, cuda_device=0, args=train_args)
    # model = ClassificationModel('distilbert', 'outputsDistilBERT20',  use_cuda=True, cuda_device=2, args=train_args)

    hasocDf = pd.read_csv('./DataSets/HASOC/preprocessed/HASOC_Train_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    print('data read', hasocDf.head(), hasocDf.shape)
    labels = []
    for row in hasocDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(1)
        else:
            labels.append(0)
    print(labels[:5])
    hasocDf['labelsEncoded'] = labels
    print(hasocDf.head())
    # sys.exit()
    # print('Train the model')
    # train_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[:-500], hasocDf['labelsEncoded'].values.tolist()[:-500])), columns=['tweets', 'labelsEncoded'])
    # eval_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[-500:], hasocDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    # print('head train', train_df.head(10))
    # # sys.exit()
    # model.train_model(train_df, eval_df=eval_df)
    


def trainHasoc():
    train_args= {
    "output_dir": "outputsHasocOlid", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }

    print(os.getcwd())
    model = ClassificationModel('roberta', 'roberta-base',num_labels=2, use_cuda=True, cuda_device=0, args=train_args)
    # model = ClassificationModel('distilbert', 'outputsDistilBERT20',  use_cuda=True, cuda_device=2, args=train_args)
    print('model?***************************************************************', model.__dict__)
    hasocDf = pd.read_csv('DataSets/HASOC/preprocessed/HASOC_Train_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    print(hasocDf.head(), hasocDf.shape)
    labels = []
    for row in hasocDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    hasocDf['labelsEncoded'] = labels
    print(hasocDf.head())

    print('Train the model')
    train_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[:-500], hasocDf['labelsEncoded'].values.tolist()[:-500])), columns=['tweets', 'labelsEncoded'])
    eval_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[-500:], hasocDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    print(train_df.head(), 'head tr', train_df.shape)
    print(eval_df.head(), 'eval', eval_df.shape)
    # sys.exit()
    model.train_model(train_df, eval_df=eval_df)

def f1_macro(y_true,y_pred):
    return f1_score(y_true,y_pred,average='macro')


def evalHasoc():
    train_args= {
    "output_dir": "outputsClassHASOC/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False, #when doing regression to True from False
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }

    hasocDf = pd.read_csv('DataSets/HASOC/preprocessed/HASOC_Test_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    print(hasocDf.head(), hasocDf.shape)
    labels = []
    for row in hasocDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(np.int(1)) #when doing regression to float from long
        else:
            labels.append(np.int(0)) #when doing regression to float from long
    print(labels[:5], type(labels[0]))
    hasocDf['labelsEncoded'] = labels
    print(hasocDf.head())

    print('Test the model')
    test_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist(), hasocDf['labelsEncoded'].values.tolist())), columns=['tweets', 'labelsEncoded'])
    # eval_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[-500:], hasocDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    print(test_df.head())
                                            #change this depending on the model #when doing regression this needt to say one num_labels
    model = ClassificationModel('roberta', 'outputsFive4', num_labels=2, use_cuda=True, cuda_device=0, args=train_args)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_macro)
    print(result)
    print(model_outputs)
    # print(wrong_predictions)

def predictForOlidpre():
    train_args= {
    "output_dir": "outputsClass/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': True, #when doing regression to True from False
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }

    olidDF = pd.read_csv('./DataSets/NewOLIDfiles/OLID_TEST_A_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    olidLabelsA = pd.read_csv('./DataSets/NewOLIDfiles/labels-levela.csv', delimiter=',')
    olidDF['LabelsA'] = olidLabelsA['label'].values.tolist()
    print(olidDF.head(), olidDF.shape)
    print(olidLabelsA.head())
    # sys.exit()
    labels = []
    # sentencesForPrediction = [] 
    # for row in hasocDf.itertuples():
    #     # print('row[3]', row[3])
    #     if row[3] == 'HOF':
    #         labels.append(np.float(1)) #when doing regression to float from long
    #     else:
    #         labels.append(np.float(0)) #when doing regression to float from long
    # print(labels[:5], type(labels[0]))
    # hasocDf['labelsEncoded'] = labels
    # print(hasocDf.head())

    # print('Test the model')
    # test_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist(), hasocDf['labelsEncoded'].values.tolist())), columns=['tweets', 'labelsEncoded'])
    # # eval_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[-500:], hasocDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    # print(test_df.head())
                                            #I have changed the name for regreesion to outputsRobertaForREgreesion instead of outputsClass
                                                        #  #when doing regression this needt to say one
    model = ClassificationModel('roberta', 'outputsClass', num_labels=1, use_cuda=True, cuda_device=0, args=train_args)
    # Evaluate the model
    # result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_score)
    # print(result)
    # print(model_outputs)
    # # print(wrong_predictions)
    sentencesForPrediction = []
    compareLabels = []
    for row in olidDF.itertuples():
        # sentencesForPrediction.append([row[2]])
        # predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        # print('row[0]', row[1],'row[2]', row[2])
        # print(row)
        # sys.exit()
        predictions, raw_outputs = model.predict([row[2]])#[["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        # print('pred', prections, predictions > 0.1)
        print('raw', raw_outputs)
        sentencesForPrediction.append((row[2], raw_outputs, row[3]))
        # print(sentencesForPrediction)
        # sys.exit()
    print('results', sentencesForPrediction)
    # predictions, raw_outputs = model.predict(sentencesForPrediction)#[["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
    # print('len', len(raw_outputs), raw_outputs)
    # for elem in sentencesForPrediction:
    #     print(elem)
    #     compareLabels.append(elem
    olidDF['LabelsTaskACompare'] = sentencesForPrediction
    outputToPandas = pd.DataFrame(sentencesForPrediction)
    print(outputToPandas.head())
    outputToPandas.to_csv('./DataSets/NewOLIDfiles/OlidPredictionsTest.csv', sep='\t')
    print(olidDF.head(50))

def trainOnBothOlidHASOC():
    train_args= {
    "output_dir": "outputsBothOlidHASOC", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }

    print(os.getcwd())
    model = ClassificationModel('roberta', 'outputsRobertaOlidOnly',num_labels=2, use_cuda=True, cuda_device=0, args=train_args)
    # model = ClassificationModel('distilbert', 'outputsDistilBERT20',  use_cuda=True, cuda_device=2, args=train_args)
    print('model?***************************************************************', model.__dict__)
    hasocDf = pd.read_csv('DataSets/HASOC/preprocessed/HASOC_Train_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    print(hasocDf.head(), hasocDf.shape)
    labels = []
    for row in hasocDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    hasocDf['labelsEncoded'] = labels
    print(hasocDf.head())

    print('Train the model')
    train_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[:-500], hasocDf['labelsEncoded'].values.tolist()[:-500])), columns=['tweets', 'labelsEncoded'])
    eval_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[-500:], hasocDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    print(train_df.head(), 'head tr', train_df.shape)
    print(eval_df.head(), 'eval', eval_df.shape)
    # sys.exit()
    model.train_model(train_df, eval_df=eval_df)

def trainOlidOnly():
    train_args= {
    "output_dir": "outputsRobertaOlidOnly/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every eåoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 3, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 2000,
    "evaluate_during_training_verbose": True,
    }

    print('Create a ClassificationModel/regression')

    # model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=1, use_cuda=True, cuda_device=0, args=train_args) #first
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, use_cuda=True, cuda_device=0, args=train_args)
    # print(train_df.head())
    olidDf = pd.read_csv('DataSets/NewOLIDfiles/OLID_Tain_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    print(olidDf.head(), olidDf.shape)
    labels = []
    for row in olidDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'OFF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    olidDf['labelsEncoded'] = labels
    print(olidDf.head())

    print('Train the model')
    train_df = pd.DataFrame(list(zip(olidDf['tweet'].values.tolist()[:-500], olidDf['labelsEncoded'].values.tolist()[:-500])), columns=['tweets', 'labelsEncoded'])
    eval_df = pd.DataFrame(list(zip(olidDf['tweet'].values.tolist()[-500:], olidDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    print(train_df.head(), 'head tr', train_df.shape)
    print(eval_df.head(), 'eval', eval_df.shape)
    print('Train the model')
    model.train_model(train_df, eval_df=eval_df)

def evalOlidOnly():
    train_args= {
    "output_dir": "outputsRobertaOlidOnly/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every eåoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 3, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 2000,
    "evaluate_during_training_verbose": True,
    }


    olidTestDf = pd.read_csv('DataSets/NewOLIDfiles/OLID_TEST_A_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')#, nrows=3310)
    olidLabelsA = pd.read_csv('DataSets/NewOLIDfiles/labels-levela.csv', delimiter=',')
    print(olidTestDf.head())
    print(olidTestDf.shape)
    print(olidLabelsA.head())
    print(olidLabelsA.shape)
    olidTestDf['Labels'] = olidLabelsA['label']
    print(olidTestDf.head(10))
    print(olidTestDf['tweet'])
    
    labels = []
    for row in olidTestDf.itertuples():
        print('row[3]', row)
        # sys.exit()
        if row[3] == 'OFF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    olidTestDf['labelsEncoded'] = labels
    print(olidTestDf.head())

    model = ClassificationModel('roberta', 'outputsRobertaOlidOnly', num_labels=2, use_cuda=True, cuda_device=0, args=train_args)

    test_df = pd.DataFrame(list(zip(olidTestDf['tweet'].values.tolist(), olidTestDf['labelsEncoded'].values.tolist())), columns=['tweets', 'labelsEncoded'])
    # eval_df = pd.DataFrame(list(zip(hasocDf['tweet'].values.tolist()[-500:], hasocDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    print(test_df.head())

    result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_score)
    print(result)
    print(model_outputs)

def trainOnBothHASOCOlid():
    train_args= {
    "output_dir": "outputsHasocOlid", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }

    print(os.getcwd())
    model = ClassificationModel('roberta', 'outputsHasocOlid',num_labels=2, use_cuda=True, cuda_device=1, args=train_args)
    # model = ClassificationModel('distilbert', 'outputsDistilBERT20',  use_cuda=True, cuda_device=2, args=train_args)
    print('model?***************************************************************', model.__dict__)
    olidDf = pd.read_csv('DataSets/NewOLIDfiles/OLID_Tain_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    print(olidDf.head(), olidDf.shape)
    labels = []
    for row in olidDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'OFF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    olidDf['labelsEncoded'] = labels
    print(olidDf.head())

    print('Train the model')
    train_df = pd.DataFrame(list(zip(olidDf['tweet'].values.tolist()[:-500], olidDf['labelsEncoded'].values.tolist()[:-500])), columns=['tweets', 'labelsEncoded'])
    eval_df = pd.DataFrame(list(zip(olidDf['tweet'].values.tolist()[-500:], olidDf['labelsEncoded'].values.tolist()[-500:])), columns=['tweets', 'labelsEncoded'])
    print(train_df.head(), 'head tr', train_df.shape)
    print(eval_df.head(), 'eval', eval_df.shape)
    print('Train the model')
    # sys.exit()
    model.train_model(train_df, eval_df=eval_df)

def fiveFoldExperiments(fold=0, dataset='hasoc'):
    train_args= {
    "output_dir": "outputsFive"+str(fold), # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }
    hasocTrainFiles = ['HASOC8020_Train1_ATUSER_URL_EmojiRemoved_Pedro.txt', 'HASOC8020_Train2_ATUSER_URL_EmojiRemoved_Pedro.txt', 
                  'HASOC8020_Train3_ATUSER_URL_EmojiRemoved_Pedro.txt', 'HASOC8020_Train4_ATUSER_URL_EmojiRemoved_Pedro.txt',
                  'HASOC8020_Train5_ATUSER_URL_EmojiRemoved_Pedro.txt']
    hasocEvalFiles = ['HASOC8020_Dev1_ATUSER_URL_EmojiRemoved_Pedro.txt', 'HASOC8020_Dev2_ATUSER_URL_EmojiRemoved_Pedro.txt', 
                      'HASOC8020_Dev3_ATUSER_URL_EmojiRemoved_Pedro.txt', 'HASOC8020_Dev4_ATUSER_URL_EmojiRemoved_Pedro.txt',
                      'HASOC8020_Dev5_ATUSER_URL_EmojiRemoved_Pedro.txt' ]
    olidTrainFiles = ['OLID_HASOC8020_Train1_ATUSER_URL_EmojiRemoved_Pedro.txt', 'OLID_HASOC8020_Train2_ATUSER_URL_EmojiRemoved_Pedro.txt',
                      'OLID_HASOC8020_Train3_ATUSER_URL_EmojiRemoved_Pedro.txt', 'OLID_HASOC8020_Train4_ATUSER_URL_EmojiRemoved_Pedro.txt', 
                      'OLID_HASOC8020_Train5_ATUSER_URL_EmojiRemoved_Pedro.txt']
    
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, use_cuda=True, cuda_device=0, args=train_args)
    print(fold, hasocTrainFiles[fold])
    hasocTrainDf = pd.read_csv('DataSets/FiveFold/'+hasocTrainFiles[fold], delimiter='\t')
    hasocEvalDf = pd.read_csv('DataSets/FiveFold/'+hasocEvalFiles[fold], delimiter='\t')

    olidTrainDf = pd.read_csv('DataSets/FiveFold/'+olidTrainFiles[fold], delimiter='\t')

    labels = []
    for row in hasocTrainDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    hasocTrainDf['labelsEncoded'] = labels
    print(hasocTrainDf.head())

    labels = []
    for row in hasocEvalDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    hasocEvalDf['labelsEncoded'] = labels
    print(hasocEvalDf.head())

    labels = []
    for row in olidTrainDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(np.int(1))
        else:
            labels.append(np.int(0))
    print(labels[:5], type(labels[0]))
    olidTrainDf['labelsEncoded'] = labels
    print(olidTrainDf.head())

    if dataset == 'hasoc':
        print('Train the model')
        train_df = pd.DataFrame(list(zip(hasocTrainDf['tweet'].values.tolist(), hasocTrainDf['labelsEncoded'].values.tolist())), columns=['tweets', 'labelsEncoded'])
        eval_df = pd.DataFrame(list(zip(hasocEvalDf['tweet'].values.tolist(), hasocEvalDf['labelsEncoded'].values.tolist())), columns=['tweets', 'labelsEncoded'])
        print(train_df.head(), 'head tr', train_df.shape)
        print(eval_df.head(), 'eval', eval_df.shape)

        print('*********')
        print(hasocTrainDf.head())
        # sys.exit()
    elif dataset == 'olid':
        print('Train the model elif olid**********************************************************************')
        train_df = pd.DataFrame(list(zip(olidTrainDf['tweet'].values.tolist(), olidTrainDf['labelsEncoded'].values.tolist())), columns=['tweets', 'labelsEncoded'])
        eval_df = pd.DataFrame(list(zip(hasocEvalDf['tweet'].values.tolist(), hasocEvalDf['labelsEncoded'].values.tolist())), columns=['tweets', 'labelsEncoded'])
        print(train_df.head(), 'head tr', train_df.shape)
        print(eval_df.head(), 'eval', eval_df.shape)
        print('Train the model')
    else:
        print('Wrong training selected')
        sys.exit()
    model.train_model(train_df, eval_df=eval_df)

def predictHasoc(fold=0):
    train_args= {
    #"output_dir": "outputsClassHASOC/", # save things
    "cache_dir": "cache/", #cache dir
    "save_model_every_epoch": True, #every epoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 1, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': False, #when doing regression to True from False
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    }

    hasocDf = pd.read_csv('DataSets/HASOC/preprocessed/HASOC_Test_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
    print(hasocDf.head(), hasocDf.shape)
    labels = []
    for row in hasocDf.itertuples():
        # print('row[3]', row[3])
        if row[3] == 'HOF':
            labels.append(np.int(1)) #when doing regression to float from long
        else:
            labels.append(np.int(0)) #when doing regression to float from long
    print(labels[:5], type(labels[0]))
    hasocDf['labelsEncoded'] = labels
    print(hasocDf.head())

    # model = ClassificationModel('roberta', f'outputsFive{fold}', num_labels=2, use_cuda=True, cuda_device=0, args=train_args)


    sentencesForPrediction = []
    compareLabels = []
    for row in hasocDf.itertuples():
        # sentencesForPrediction.append([row[2]])
        # predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        print('row[1]:', row[1],'row[2]:', row[2])
        pred, rawOutputs = model.predict([row[2]])#[["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
        print('pred', pred, pred > 0.1)
        print('raw', rawOutputs)
        # if rawOutputs > 0.5:
        sentencesForPrediction.append((row[1], row[3], rawOutputs))
        # else:
            # sentencesForPrediction.append((row[1], row[3], raw_outputs, 'NOT'))

        # print(sentencesForPrediction)
        # sys.exit()
    print('results', sentencesForPrediction)
    # predictions, raw_outputs = model.predict(sentencesForPrediction)#[["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
    # print('len', len(raw_outputs), raw_outputs)
    # for elem in sentencesForPrediction:
    #     print(elem)
    #     compareLabels.append(elem
    # dataframe['LabelsTaskACompare'] = sentencesForPrediction
    outputToPandas = pd.DataFrame(sentencesForPrediction)
    print(outputToPandas.head())
    outputToPandas.to_csv(f'./compareOut/outputsFoldOlidHasoc{fold}.csv', sep='\t')
    print(hasocDf.head())
    # model = ClassificationModel('roberta', 'outputsRobertaOlidOnly', num_labels=2, use_cuda=True, cuda_device=0, args=train_args)
    # Evaluate the model
    # result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_score)
    # print(result)
    # print(model_outputs)
    # print(wrong_predictions)
    # prediction, rawOutputs = model.predi

def trainModelsParallel(num = 0):

    train_args= {
    "output_dir": f"outputs6MilRobertaChunks{num}", # save things
    "cache_dir": f"cache{num}", #cache dir
    "save_model_every_epoch": True, #every eåoch save it
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "save_eval_checkpoints": False,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    'num_train_epochs': 20,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    # "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 1e-5, #was 4e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    # "num_train_epochs": 50,
    # "train_batch_size": 24,
    "n_gpu": 3, #use several gpus
    "manual_seed": 42, #reproducible results
    'regression': True,
    "save_eval_checkpoints": False,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 2000,
    "evaluate_during_training_verbose": True,
    }

    print('Create a ClassificationModel/regression')

    # model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=1, use_cuda=True, cuda_device=0, args=train_args) #first
    model = ClassificationModel('roberta', 'roberta-base', num_labels=1, use_cuda=True, cuda_device=0, args=train_args)
    # print(train_df.head())

    print('Train the model')
    model.train_model(train_df_chunks[num], eval_df=eval_df)

    # Evaluate the model
    # result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    # print('result', result)
    # print('wrong_prediction', wrong_predictions)
    # predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
    # print('predictions...')
    # print(predictions)
    # print(raw_outputs)


# trainModel()


loadOlidOffenseEvalData()

# findCutoff()
print('eval time')
# evalTestSet()
print('retrain time')
# retrainModel()
# trainHasoc()
print('eval hasoc')
# evalHasoc()
# retrainModel()
# predictForOlidpre()
# trainOlidOnly()
# evalOlidOnly()
# trainOnBothOlidHASOC()
# trainOnBothHASOCOlid()
# fiveFoldExperiments(4, 'olid')
# predictHasoc(4)
# trainModelsParallel(4)