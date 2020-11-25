# -*- coding: utf-8 -*-
"""
Created on Sep 13 11:24:41 2017

The project used machine learning methods to classify the input project evaluation documents into 8 pre-defined lessons categories (46 sub-categories), based on World Bank IEG project text data

Note for Users
    * Put the raw word documents that you want to evaluate under the folder "input_doc"
    * Run the program, and specify the methods names and expected output categories (8 or 46)

@author: Yuan Xiang
"""

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import re
import pickle
import os
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')

# Common path
strPath = "/Users/yuanxiang/Documents/Work/Text-Category/project-lesson-category/"
# Input documents path
folderPath=os.path.join(strPath, input_doc)

##### PART 1 Use pre-trained model #####
# load and clean data
lesson_raw = pd.read_csv(os.path.join(strPath, "data", 'training_corpus.csv'), encoding = "ISO-8859-1")

lesson_raw = lesson_raw.dropna(axis = 0, how="any")

X = lesson_raw['Text']

Y = lesson_raw['Category']
# merge sub-categories into categories
Y1 = Y.copy()
Y1[(Y1 >= 1) & (Y1 <= 10)] = 1
Y1[(Y1 >= 11) & (Y1 <= 15)] = 2
Y1[(Y1 >= 16) & (Y1 <= 22)] = 3
Y1[(Y1 >= 23) & (Y1 <= 33)] = 4
Y1[(Y1 >= 34) & (Y1 <= 43)] = 5
Y1[(Y1 >= 44) & (Y1 <= 49)] = 6
Y1[(Y1 >= 50) & (Y1 <= 54)] = 7
Y1[(Y1 >= 55) & (Y1 <= 64)] = 8
Y1[(Y1 >= 65) & (Y1 <= 71)] = 8
Y1[(Y1 >= 72) & (Y1 <= 74)] = 6
Y1[(Y1 >= 75) & (Y1 <= 81)] = 7

Category = Y.copy()
# Get user input for parameters

# assign input category to different scenarios
while True:
    Model = input('Which model do you want to use, NB, RF or NN: ').lower()
    Category_indicator = input('How many output categories do you want, 8 or 46: ')
    try:
        if Category_indicator == '8':
            Category = Y1.copy()
            break
        if Category_indicator == '46':
            Category = Y.copy()
            break
    except:
        print('You have to input either 8 or 64 !')
        break

# set parameters
cv_num = 0
test_size = 0.2
seed = 1

# load new data
# find them
## document pre-process
def prepare_documents(folderPath=folderPath):

    # initialize output
    paragraph_list = []

    # get all the documents
    for file in os.listdir(folderPath):
        if file.endswith(".docx"):
            try:
                document_id = re.split('_|\s', os.path.splitext(file)[0])[0]
                document = Document(file)

                # find paragraphs
                for index, paragraph in enumerate(document.paragraphs):

                    # get the tokenized words
                    temp_word_list = re.sub("[^\w]", " ", paragraph.text).split()
                    # clean data(save sentences whose length is greater than 20)
                    if paragraph.text not in '' and len(temp_word_list) >= 10:
                        paragraph_list.append((document_id + '_' + str(index), paragraph.text))
            except:
                pass
    paragraph_df = pd.DataFrame(paragraph_list, columns=['id', 'text'])
    return paragraph_df

# save the loaded documents to a dataframe
df = prepare_documents()
X_new = df['text']

# load customized stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))
cust_stopwords = pd.read_csv(strPath + 'stopwords.csv', header = None)


stopwords.update(set(cust_stopwords[0]))


folderPath = os.path.join(os.getcwd(),'documents')


## Tokenize
# Tokenize with stemming
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        token = re.sub('[^\w+]', "", token)
        if re.search('[a-zA-Z]', token):
            if token not in stopwords:
                filtered_tokens.append(token)
    stems = [PorterStemmer().stem(word) for word in filtered_tokens]
    return stems

# Tokenize without stemming
def tokenize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        token = re.sub('[^\w+]', "", token)
        if re.search('[a-zA-Z]', token):
            if token not in stopwords:
                filtered_tokens.append(token)
    return filtered_tokens

# load pre-defined keyword lists
list_category = pd.read_excel(strPath + 'features.xlsx')

keywords  = set(list_category.TERM)
keywords = [item.lower() for item in keywords]


# define vectorizer
count_vectorizer = CountVectorizer(vocabulary=keywords, ngram_range=(1,2),stop_words= stopwords,
                                   tokenizer = tokenize)
tfidf_vectorizer = TfidfVectorizer(vocabulary=keywords, ngram_range=(1,2),stop_words= stopwords,
                                   tokenizer = tokenize)

# create dtm for whole matrix
dtm = count_vectorizer.fit_transform(X_new)
tfidf_dtm = tfidf_vectorizer.fit_transform(X_new)

## start training and prediction
if Model == 'nb':
    print('Predicting using Naive Bayes Model...')
    # if not using cross validation
    ## split data into training/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Category, test_size=test_size, random_state=seed)
    X_train_dtm = count_vectorizer.fit_transform(X_train)
    X_test_dtm = count_vectorizer.transform(X_new)

    ## Naive Bayes Model
    nb_classifier = MultinomialNB()
    # nb_classifier = ComplementNB()

##################
    nb_classifier.fit(X_train_dtm, Y_train)

    # save the model
    if Category.equals(Y1):
        with open(os.path.join('model_output', 'new_nbcount_c8.sav'), 'wb') as f:
            pickle.dump(nb_classifier, f)

        with open(os.path.join('model_output', 'new_nbcount_c8.sav'), 'rb') as f:
            nb_classifier = pickle.load(f)
    else:
        with open(os.path.join('model_output', 'new_nbcount_c46.sav'), 'wb') as f:
            pickle.dump(nb_classifier, f)

        # load model
        with open(os.path.join('model_output', 'new_nbcount_c46.sav'), 'rb') as f:
            nb_classifier = pickle.load(f)

    # single prediction
    nb_prediction = nb_classifier.predict(X_test_dtm)

    # multiple prediction(topn=3) for non-continuous predictions
    nb_multi_prediction = nb_classifier.predict_proba(X_test_dtm)
    nb_pred_top3 = pd.DataFrame(nb_classifier.classes_[(-np.array(nb_multi_prediction)).argsort()]).loc[:, :2]
    nb_df_result = pd.concat([X_new.reset_index(), nb_pred_top3], axis=1)
    mask = (nb_df_result.iloc[:, 1] == nb_df_result.iloc[:, 2]) | (
    nb_df_result.iloc[:, 1] == nb_df_result.iloc[:, 3])
    print("Test Done")
 
    # save the prediction
if Category.equals(Y1):
        nb_df_result.to_excel(os.path.join('tables', 'nb_prediction_c8_p3.xlsx'))
else:
        nb_df_result.to_excel(os.path.join('tables', 'nb_prediction_c46_p3.xlsx'))
    
print('Finish prediction with Naive Bayes Model !')
    # return nb_score

# For Random Forest Model
if Model == 'rf':
    print('Running Random Forest Model...')
    # if not using cross validation
    ## split data into training/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Category, test_size=test_size, random_state=seed)
    X_train_dtm = count_vectorizer.fit_transform(X_train)
    X_test_dtm = count_vectorizer.transform(X_new)

    ## Random Forest Model
    forest_classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
    forest_classifier.fit(X_train_dtm, Y_train)

    # save the model
    if Category.equals(Y1):
        with open(os.path.join('model_output', 'new_rfcount_c8.sav'), 'wb') as f:
            pickle.dump(forest_classifier, f)

        with open(os.path.join('model_output', 'new_rfcount_c8.sav'), 'rb') as f:
            nb_classifier = pickle.load(f)
    else:
        with open(os.path.join('model_output', 'new_rfcount_c46.sav'), 'wb') as f:
            pickle.dump(forest_classifier, f)

        # load model
        with open(os.path.join('model_output', 'new_rfcount_c46.sav'), 'rb') as f:
            nb_classifier = pickle.load(f)

    # single prediction
    forest_prediction = forest_classifier.predict(X_test_dtm)

    # multiple prediction(topn=3) for non-continuous predictions
    forest_multi_prediction = forest_classifier.predict_proba(X_test_dtm)
    forest_pred_top3 = pd.DataFrame(
        forest_classifier.classes_[(-np.array(forest_multi_prediction)).argsort()]).loc[:, :2]
    forest_df_result = pd.concat([X_new.reset_index(), forest_pred_top3], axis=1)
    mask = (forest_df_result.iloc[:, 1] == forest_df_result.iloc[:, 2]) | (
        forest_df_result.iloc[:, 1] == forest_df_result.iloc[:, 3])

# For Neural Network
# Training artificial neural network
if Model == "nn":
    print('Running Neural Networks Model...')
    # if not using cross validation
    ## split data into training/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Category, test_size=test_size, random_state=seed)
    X_train_dtm = count_vectorizer.fit_transform(X_train)
    X_test_dtm = count_vectorizer.transform(X_new)

    num_hidden_neurons = 10000
    clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu',
                        hidden_layer_sizes=(num_hidden_neurons,), random_state=1)
    clf.fit(X_train_dtm, Y_train)

    # save the model
    if Category.equals(Y1):
        with open(os.path.join('model_output', 'new_nncount_c8.sav'), 'wb') as f:
            pickle.dump(clf, f)

        with open(os.path.join('model_output', 'new_nncount_c8.sav'), 'rb') as f:
            nb_classifier = pickle.load(f)
    else:
        with open(os.path.join('model_output', 'new_nncount_c46.sav'), 'wb') as f:
            pickle.dump(clf, f)

        # load model
        with open(os.path.join('model_output', 'new_nncount_c46.sav'), 'rb') as f:
            nb_classifier = pickle.load(f)

    # single prediction
    clf_prediction = clf.predict(X_test_dtm)

    # multiple prediction(topn=3) for non-continuous predictions
    clf_multi_prediction = clf.predict_proba(X_test_dtm)
    clf_pred_top3 = pd.DataFrame(
        clf.classes_[(-np.array(clf_multi_prediction)).argsort()]).loc[:, :2]
    clf_df_result = pd.concat([X_new.reset_index(), clf_pred_top3], axis=1)
    mask = (clf_df_result.iloc[:, 1] == clf_df_result.iloc[:, 2]) | (
        clf_df_result.iloc[:, 1] == clf_df_result.iloc[:, 3])

    # save the prediction
    if Category.equals(Y1):
        clf_df_result.to_excel(os.path.join('tables', 'nn_prediction_c8_p3.xlsx'))
    else:
        clf_df_result.to_excel(os.path.join('tables', 'nn_prediction_c46_p3.xlsx'))

    print('Finish prediction with Neural Network Model !')


##### END OF PART 1 #####


##### PART 2 Prediction Table Summary

document_id = df['id'].str.split('_').str.get(0)
sent_id = df['id'].str.split('_').str.get(1)

df = pd.concat([document_id, sent_id, df], axis=1)

# append prediction to dataframe
if Model == 'nb':
    df = pd.concat([df,nb_df_result.loc[:, [0,1,2]]], axis=1)
if Model == 'rf':
    df = pd.concat([df, forest_df_result.loc[:, [0,1,2]]], axis=1)
if Model == 'nn':
    df = pd.concat([df,clf_df_result.loc[:, [0,1,2]]], axis=1)

df.columns = ['project_id', 'sent_id', 'proj_sent_id', 'text', 'category1', 'category2', 'category3']


# get distribution of universe
df_universe = pd.read_excel(os.path.join('tables', 'universe_prediction.xlsx'), sheet_name='frequency')

result_list = []
# get summary for each document
for category in list(df.columns[4:7]):

    frequency = lambda x: x[category].value_counts()

    df_summary = df.loc[:, ['project_id', category]].groupby('project_id').apply(frequency).reset_index()
    df_summary.columns = ['project_id', 'category', 'frequency']
    df_summary_frequency = df_summary[['project_id', 'frequency']].groupby(['project_id']).transform(lambda x: x/x.sum())
    df_summary = pd.concat([df_summary, df_summary_frequency], axis=1)
    df_summary.columns = ['project_id', 'category', 'frequency', 'proportion']


    # join two dataframes
    df_master = pd.merge(df_summary, df_universe, on='category').drop('count', axis=1)
    df_master['salient'] = df_master['proportion_x']/df_master['proportion_y']

    # calculate most salient value within group(group by project)

    # remove the maximum value
    df_result = df_master.groupby('project_id').apply(lambda x: x.loc[x.salient !=x.salient.max()])\
        .drop('project_id', axis=1).reset_index(level=0)
    df_result = df_result.groupby('project_id').apply(lambda x: x[x.salient>=0.7*x.salient.mean()])\
        .drop('project_id', axis=1).reset_index(level=0)

    # order by salient values
    df_result = df_result.groupby('project_id')\
        .apply(lambda x: x.sort_values('salient', ascending=False)).drop('project_id', axis=1).reset_index(level=0)

    result_list.append(df_result)

# output to excel
writer = pd.ExcelWriter(os.path.join('tables', 'document_prediction.xlsx'))
df.to_excel(writer, sheet_name='project_prediction')
result_list[0].to_excel(writer, sheet_name='category1_salient_prediction')
result_list[1].to_excel(writer, sheet_name = 'category2_salient_prediction')
result_list[2].to_excel(writer, sheet_name = 'category3_salient_prediction')
# df_result_topic.to_excel(writer, sheet_name='salient_prediction_by_topic')
writer.save()

