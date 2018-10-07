# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 23:15:10 2017

@author: hongx
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans


#read all data again
df = pd.read_csv('Twitter_output_cleaned.csv' , sep=',', encoding='latin1')


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic " + str(topic_idx))
        #print (topic.argsort())
        print (" "+str([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

documents = df['text'].tolist()

no_features = 1000

# tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(df['text'].values.astype('U'))

tfidf_feature_names = tfidf_vectorizer.get_feature_names()



num_clusters = 10
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()


df['topic']=clusters


s=  pd.read_csv('sentiment_LinearSVC.txt' , sep=',', encoding='latin1')

sentimental = [0]
for i in range(len(s['0'])):
    ##print(s.loc[i,'0'])
    sentimental.append(s.loc[i,'0'])
    
df["sentimental"] = sentimental

df["sentimental"].groupby(df['topic']).mean()

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
#print(order_centroids)




for i in range(num_clusters):
    print()
    print("Cluster",i, "words: ")
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster		
        print(tfidf_feature_names[ind]," ", end='')
print()












