import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sys
import pprint as pp

print('imports completed')
olidDf = pd.read_csv('./DataSets/HASOC/preprocessed/HASOC_Train_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t') \
# olidDf = pd.read_csv('./DataSets/HASOC/preprocessed/HASOC_Test_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
# olidDf = pd.read_csv('./DataSets/NewOLIDfiles/OLID_Tain_ATUSER_URL_EmojiRemoved_Pedro.txt', delimiter='\t')
print(olidDf.head(), olidDf.shape)
labels = []
for row in olidDf.itertuples():
    # print('row[3]', row[3])
    # sys.exit()
    if row[3] == 'OFF':
        labels.append(1)
    else:
        labels.append(0)
print(labels[:5])
olidDf['labelsEncoded'] = labels
print(olidDf.head())
def createModel(df):
    model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-base')# no uncased for roberta

    ## Want BERT instead of distilBERT? Uncomment the following line:
    #model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    print('Model roberta created')

    tokenized = olidDf['tweet'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    print('text tokenized')
    print(tokenized)
    # sys.exit()

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    print(np.array(padded).shape)
    print('padded text')
    pad_df = pd.DataFrame(zip(olidDf['id'], padded, olidDf['labelsEncoded']), dtype='float64')
    pad_df.to_csv('./DataSets/NewOLIDfiles/FeaturesCompiledTrainHasocDf.csv')
    sys.exit()
    attention_mask = np.where(padded != 0, 1, 0)
    print(attention_mask.shape)
    print('attention mask added')

    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)
    print('Running data tru model')
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    print('Model ran')

    print('Gathering features and saving ')
    features = last_hidden_states[0][:,0,:].numpy()
    featuresDf = pd.DataFrame(zip(olidDf['id'], features, olidDf['labelsEncoded']), dtype='float64')
    print(featuresDf.head())
    # print(type(featuresDf['2']))
    # sys.exit()
    featuresDf.to_csv('./DataSets/NewOLIDfiles/FeaturesCompiledTrainolidDf.csv')
    featuresDf.to_pickle('./DataSets/NewOLIDfiles/FeaturesCompiledTrainolidDf.pkl')
    #  dfFeaturesSave = pd.DataFrame(zip(features, df[1]))
    # dfFeaturesSave.to_csv(f'./FeaturesCompilation/FeaturesAndLabelsForTask{task}BatchStart{batchStart}BatchEnd{batchEnd}.csv')
    print('Features saved to csv & pickle')


def loadClassificationFeatures():
    hasocDfTest = pd.read_pickle('./DataSets/HASOC/FeaturesCompiledTest.pkl')#, encoding='utf-8')
    hasocDfTrain = pd.read_pickle('./DataSets/HASOC/FeaturesCompiledTrain.pkl')#, encoding='utf-8')
    pp.pprint(('************************************* lr************', hasocDfTest.head(), hasocDfTrain.shape))
    print(hasocDfTest.isna().sum())
    print(hasocDfTrain.isna().sum())
    # sys.exit()
    labelsTrain = hasocDfTrain[2].to_numpy(dtype=int)
    labelsTest = hasocDfTest[2].to_numpy(dtype=int)
    # features = hasocDf['1'].to_numpy()
    # print(features)
    # sys.exit()
    # featuresTest = hasocDfTest[1].to_numpy()
    # featuresTrain = hasocDfTrain[1].values
    featuresTrain = []
    featuresTest = []
    print(type(featuresTrain))
    print('shape', featuresTrain)
    # sys.exit()
    for row in hasocDfTest.itertuples():
        # print(row[3])
        featuresTest.append(np.array(row[2]))
        # sys.exit()
    for row in hasocDfTrain.itertuples():
        # print(row)
        featuresTrain.append(np.array(row[2]))
        # pp.pprint(np.array(row[2]))
        # sys.exit()
    # features = hasocDf['1'].str.strip().values
    # print(featuresTest[:5])
    # sys.exit()
    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    print('train test split done')
    print(len(featuresTrain), type(labelsTrain))
    # sys.exit()
    print('training lr')
    # print(np.asmatrix(featuresTrain).shape)
    # sys.exit()
    lr_clf = LogisticRegression()
    lr_clf.fit(featuresTrain, labelsTrain)
    print('training lr done')
    # sys.exit()
    print('testing lr')
    print(lr_clf.score(featuresTest, labelsTest))
    print('all done')

# loadClassificationFeatures()
createModel(olidDf)