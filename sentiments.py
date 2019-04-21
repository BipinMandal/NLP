import spacy
import numpy as np
import re
import pandas as pd

#import train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


#remove label column from train data and merge with test data
train_content = train_data.iloc[:,[0,2]].values
train_content = pd.DataFrame(train_content, columns = ['id', 'tweet'])
train_label = train_data.iloc[:,1].values
dataset_merge = pd.concat([train_content,test_data], ignore_index= True)

#clean and preprocessing data
import wordninja as wn
from nltk.corpus import stopwords 
nlp = spacy.load('en_core_web_md',disable=['parser', 'tagger', 'ner'])
stop_words = set(stopwords.words('english'))
for index, row in dataset_merge.iterrows():
    text =  re.sub(r'((?mi)https?:[\w\/._-]*)', ' ', str(row.tweet))
    text =  re.sub(r'(?mi)\S*@\S*\s?', ' ', text)
    text =  re.sub(r'(?mi)\S*.com\S*\s?', ' ', text)
    text =  re.sub(r'(?mi)(can\'t|couldn\'t|should\'t|won\'t|arn\'t|wasn\'t|wern\'t|dont|cant)','not', text)
    text =  re.sub(r'(?mi)[^\w#]', ' ', text)
    text =  re.sub(r'(?mi)[#]', '', text)
    text =  re.sub(r'(?mi)[\d]', ' ', text)
    doc = nlp(str(text))
    doc = set([token.lemma_ for token in doc])
    doc = [token.strip() for token in doc]
    doc = [wn.split(str(token)) for token in doc]
    doc = [item for subtoken in doc for item in subtoken]
    doc = set([token for token in doc if (len(token)>3)])
    doc = ' '.join([w.lower() for w in doc if not w in stop_words ])
    dataset_merge.loc[index, 'tweet'] = doc
    
from sklearn.feature_extraction.text import CountVectorizer    
cv = CountVectorizer()
corpus = [sent for sent in dataset_merge.tweet]
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X) 
#entire data is again divided into its train and test data as it was previously arranged with id.
X_train = X.iloc[0:7920,:].values
Y_train = train_label
X_test = X.iloc[7920:9873,:].values

#train model using MultinomialNB
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, Y_train)    
y_pred = model.predict(X_test) 
y_pred = pd.Series(y_pred)   
y_pred.to_excel('y_pred.xlsx')

    
    
    
    
    
    
