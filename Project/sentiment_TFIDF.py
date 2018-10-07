#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:33:40 2017
This python file perform sentiment analysis on twitter data, and also doing some statistic
analysis using those scores predicted.
@author: 
"""

import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import re
import math
import scipy.stats as stats



Twitter_DF = pd.read_csv("Twitter_output_cleaned.csv" , sep=',', encoding='latin1')

# remove Twitter data that has date not fall in date range we want: 
# from 08/28/17 to 10/04/17
file = open('test_twitter.txt','w')
# (?:2010/12/(?:3[01]|2[5-9])|2011/01/01)
date_pattern = re.compile('(?:8/(?:3[01]|2[8-9])/17|9/(?:3[0]|2[0-9]|1[0-9]|[1-9])/17|10/(?:[1-4])/17)')
for i in Twitter_DF.index:
    date = Twitter_DF.ix[i, 'date']
    date = date[0:7]
    if bool(re.search(date_pattern, date)):
        Twitter_DF.ix[i, 'date'] = date
        file.writelines(str(Twitter_DF.ix[i, 'text']) + '\n')
    else:
        Twitter_DF = Twitter_DF.drop(i)
file.close()
    
# start doing sentiment analysis

# create a corpus for training text
corpus_train = []
with open("train_pos.txt") as train_pos:
    for line in train_pos:
        corpus_train.append(line)


with open("train_neg.txt") as train_neg:
    for line in train_neg:
        corpus_train.append(line)

# create a corpus for test text
corpus_test = []
with open("test_twitter.txt") as test_twitter:
    for line in test_twitter:
        corpus_test.append(line)

# label the training data
labels = np.zeros(1000000)
labels[0:500000]=1;
labels[500000:1000000]=0; 

totalsvm = 0           # Accuracy measure on 1000000 files
totalNB = 0
totalGSNB = 0
totalRF = 0
totalMatSvm = np.zeros((2,2))  # Confusion matrix on 1000000 files
totalMatNB = np.zeros((2,2))


# doing cross validation on training set
kf = StratifiedKFold(n_splits=10)
for train_index, test_index in kf.split(corpus_train,labels):
    X_train = [corpus_train[i] for i in train_index]
    X_test = [corpus_train[i] for i in test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # use TFIDF word 2 vector transformation
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
    
    # doing text to vector convert using TF-IDF model
    train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
    
    test_corpus_tf_idf = vectorizer.transform(X_test)
    model1 = LinearSVC()
    model2 = MultinomialNB()
    #model3 = GaussianNB()
    model4 = RandomForestClassifier()    
    model1.fit(train_corpus_tf_idf,y_train)
    model2.fit(train_corpus_tf_idf,y_train)
    #model3.fit(train_corpus_tf_idf,y_train)
    #model4.fit(train_corpus_tf_idf,y_train)
    
    result1 = model1.predict(test_corpus_tf_idf)
    result2 = model2.predict(test_corpus_tf_idf)
    #result3 = model3.predict(test_corpus_tf_idf)
    #result4 = model4.predict(test_corpus_tf_idf)
    
    
    totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1)
    totalMatNB = totalMatNB + confusion_matrix(y_test, result2)
    #totalMatGSNB = totalMatGSNB + confusion_matrix(y_test, result3)
    #totalMatRF = totalMatRF + + confusion_matrix(y_test, result4) 
    
    totalsvm = totalsvm+sum(y_test==result1)
    totalNB = totalNB+sum(y_test==result2)
    #totalGSNB = totalGSNB + sum(y_test == result3)
    #totalRF = totalRF + sum(y_test == result4)
    
print ("cross validation result: ", totalMatSvm, totalsvm/1000000.0, totalMatNB, totalNB/1000000.0 ) 


# doing sentiment prediction on our own twitter data
vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
train_tf_idf = vectorizer.fit_transform(corpus_train)
test_tf_idf = vectorizer.transform(corpus_test) 

model1 = LinearSVC()
model2 = MultinomialNB()
#model3 = GaussianNB()
#model4 = RandomForestClassifier()    
model1.fit(train_tf_idf,labels)
model2.fit(train_tf_idf,labels)
result1 = model1.predict(test_tf_idf)
result2 = model2.predict(test_tf_idf)

# output predicted results from linearSVC
with open('sentiment_LinearSVC.txt', 'w') as output:
    result_DF1 = pd.DataFrame(data = result1)
    result_DF1.to_csv(output, sep = '|', index= False)
output.close()

# output predicted results from MultinomialNB()
with open('sentiment_MultinomialNB.txt', 'w') as output:
    result_DF2 = pd.DataFrame(data = result2)
    result_DF2.to_csv(output, sep = '|', index= False)
output.close()


    

    
# change format of date from string to date   
Twitter_DF['date'] = pd.to_datetime(Twitter_DF['date']) 
Twitter_DF["sentiment_labels"] = pd.Series(result1)

# calculate mean value of number of retweets
retweet_mean = Twitter_DF['retweets'].mean() 

# calculate mean value of number of favorites
favorites_mean = Twitter_DF['favorites'].mean() 

# calculate standard deviation value of number of retweets
retweet_std = math.sqrt(Twitter_DF['retweets'].var()) 

# calculate standard deviation value of number of favorites
favorites_std = math.sqrt(Twitter_DF['favorites'].var()) 

# calculate median value of number of retweets
retweet_median = Twitter_DF['retweets'].median() 

# calculate median value of number of favorites
favorites_median = Twitter_DF['favorites'].median() 







    
    
  
group_result = Twitter_DF.groupby(['date'])['sentiment_labels'].mean()
group_mean = group_result.to_frame()






# calculate mean value of sentiment label for each day 
mean_before = group_mean['sentiment_labels'][0:15].mean() # 18 is the index of the date of Sept 13
mean_after = group_mean['sentiment_labels'][15:len(group_mean)].mean()

# calculate variance of data in column "sentiment_labels"
std_before = math.sqrt(group_mean["sentiment_labels"][0:15].var())
std_after = math.sqrt(group_mean["sentiment_labels"][15:len(group_mean)].var())   
    
# calculate median value of data in column
median_before = group_mean["sentiment_labels"][0:15].median()
median_after = group_mean["sentiment_labels"][15:len(Twitter_DF.index)].median()

# perform independent two sample t-test on mean of sentiment score before and after Apple announcement date
result = stats.ttest_ind(a= group_mean['sentiment_labels'][0:15],
                b= group_mean['sentiment_labels'][15:len(group_mean)],
                equal_var=False)
print("result after performaing independent two-sample t-test: ", result)

#group_mean["sentiment_change"] = pd.Series()
#group_mean.ix["2017-08-28", "sentiment_change"] = 0

# calculate sentiment difference between adjacent days
for i in range(len(group_mean.index)):
   if(i ==0):
       continue
   else:
       group_mean.ix[group_mean.index[i], "sentiment_change"] = group_mean.ix[group_mean.index[i], "sentiment_labels"]-group_mean.ix[group_mean.index[i-1], "sentiment_labels"] 
   
# output statistic of sentiment scores
outputFileName = "Twitter_sentiment_scores.csv"    
with open(outputFileName, 'w') as output:  
    group_mean.to_csv(output, sep = ',')
output.close()



    


