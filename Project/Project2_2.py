# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:20:55 2017

@author: hongx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score
from apyori import apriori
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc



mydf = pd.read_csv('APPLE_Stock_Clean.csv' , sep=',', encoding='latin1')
## Delete useless index
del mydf['Unnamed: 0']
    
## Calculate the price change and the volatility of 
for i in range(len(mydf)):
    try:
        mydf.loc[i,'Price Change']= (mydf.loc[i,'price']-mydf.loc[i-1,'price'])/mydf.loc[i-1,'price']*100
        mydf.loc[i,'Volatility']= np.std(mydf.loc[(i-11):i,'price'])*100
        mydf.loc[i,'Market Change']= (mydf.loc[i,'Nasdaq_price']-mydf.loc[i-1,'Nasdaq_price'])/mydf.loc[i-1,'Nasdaq_price']*100
        mydf.loc[i,'Market Volatility']= np.std(mydf.loc[(i-11):i,'Nasdaq_price'])*100
    except KeyError:
        mydf.loc[i,'Price Change']= 0
        mydf.loc[i,'Volatility']= 0
        mydf.loc[i,'Market Change']= 0
        mydf.loc[i,'Market Volatility']= 0

## Mean beta of whole market
print('Market Change')
print(np.mean(mydf['Market Change']))
print(np.median(mydf['Market Change']))
print(np.std(mydf['Market Change']))

## Median of Market Volatility
print('Market Volatility')
print(np.mean(mydf['Market Volatility']))
print(np.median(mydf['Market Volatility']))
print(np.std(mydf['Market Volatility']))

## Mean, Median, Standard Deviation of price(Could also understand as Volatility over the whole period of time)
print('price')
print(np.mean(mydf['price']))
print(np.median(mydf['price']))
print(np.std(mydf['price']))

## Mean, Median, Standard Deviation of price(Could also understand as Volatility over the whole period of time)
print('Nasdaq Price')
print(np.mean(mydf['Nasdaq_price']))
print(np.median(mydf['Nasdaq_price']))
print(np.std(mydf['Nasdaq_price']))

## Mean, Median, Standard Deviation of Volume
print("AAPL Volume")
print(np.mean(mydf['volume']))
print(np.median(mydf['volume']))
print(np.std(mydf['volume']))

## Mean, Median, Standard Deviation of AAPL Beta
print('AAPL Beta')
print(np.mean(mydf['Price Change']))
print(np.median(mydf['Price Change']))
print(np.std(mydf['Price Change']))

## Mean, Median, Standard Deviation of Market Volume
print("Nasdaq Volume")
print(np.mean(mydf['Nasdaq_volume']))
print(np.median(mydf['Nasdaq_volume']))
print(np.std(mydf['Nasdaq_volume']))



## Mean, Median, Standard Deviation of AAPL Beta
print('AAPL Volatility')
print(np.mean(mydf['Volatility']))
print(np.median(mydf['Volatility']))
print(np.std(mydf['Volatility']))


## Bing the data with the Equal depth method
mydf['AAPL_Beta_Binning']=pd.qcut(mydf['Price Change'], 5, labels=["Plunge(AC)", "Lower than Average(AC)", "Average(AC)", "Better than Average(AC)","Soar(AC)"])
mydf['Volume_Binning']=pd.qcut(mydf['volume'], 5, labels=["Small(AV)", "Small than Average(AV)", "Average(AV)", "Lager than Average(AV)","Larger(AV)"])
mydf['Nasdaq_Beta_Binning']=pd.qcut(mydf['Market Change'], 5, labels=["Plunge(MC)", "Lower than Average(MC)", "Average(MC)", "Better than Average(MC)","Soar(MC)"])
mydf['Market_Volume_Binning']=pd.qcut(mydf['Nasdaq_volume'], 5, labels=["Small(MV)", "Small than Average(MV)", "Average(MV)", "Lager than Average(MV)","Larger(MV)"])
mydf['Volatility_Binning']=pd.qcut(mydf['Volatility'], 5, labels=["Small(AVol)", "Small than Average(AVol)", "Average(AVol)", "Lager than Average(AVol)","Larger(AVol)"])
mydf['Market_Volatility_Binning']=pd.qcut(mydf['Market Volatility'], 5, labels=["Small(MVol)", "Small than Average(MVol)", "Average(MVol)", "Lager than Average(MVol)","Larger(MVol)"])

## get means by group
mydf.groupby(['AAPL_Beta_Binning']).mean()
mydf.groupby(['Nasdaq_Beta_Binning']).mean()
mydf.groupby(['Volume_Binning']).mean()
mydf.groupby(['Market_Volume_Binning']).mean()
mydf.groupby(['Volatility_Binning']).mean()
mydf.groupby(['Market_Volatility_Binning']).mean()

print(np.corrcoef(mydf['Price Change'],mydf['Volatility']))
print(np.corrcoef(mydf['price'],mydf['Nasdaq_price']))



##print(max(mydf['Price Change']))
plt.hist(mydf['Price Change'], bins=[-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.20])
plt.title("AAPL Beta Histgram")
plt.show

##Scatter plot
x = mydf["Price Change"]
y = mydf["Market Change"]

plt.scatter(x,y)
plt.show()

## Create a pure quantitative dataframe
mydf_quant = mydf.copy()
del mydf_quant['ticker']
del mydf_quant['time']
del mydf_quant['AAPL_Beta_Binning']
del mydf_quant['Volume_Binning']
del mydf_quant['timestamp']
del mydf_quant['Nasdaq_Beta_Binning']
del mydf_quant['Market_Volume_Binning']
del mydf_quant['Volatility_Binning']
del mydf_quant['Market_Volatility_Binning']


#corr = pd.Dataframe(np.corrcoef(mydf_quant))

pd.DataFrame.corr(mydf_quant)

## Output the complete correlation matrix
with open("corr.csv", 'w', encoding='latin1') as output:
    pd.DataFrame.corr(mydf_quant).to_csv(output, sep = ',', index= False)
output.close()

##Normalize the data
x = mydf_quant.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalizedDataFrame = pd.DataFrame(x_scaled,columns=['price',	'open'	, 'high',	'low','close',	'volume','vwap'	
        'Nasdaq_price','Nasdaq_volume',	'Price Change',	'Volatility',	'Market', 'Change',	'Market Volatility'])

cluster_range = range(1, 20)
cluster_errors = []
for num_clusters in cluster_range:
  clusters = KMeans(num_clusters)
  clusters.fit(x_scaled)
  cluster_errors.append(clusters.inertia_)

clusters_df = pd.DataFrame({ "num_clusters":cluster_range, "cluster_errors": cluster_errors })
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o")  
    
    
## Function for Kmeans cluster
def Kmeans(df,k_i):
    k = k_i
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(df)
        
    centroids = kmeans.cluster_centers_
       
    ##pprint(cluster_labels)
    pprint(centroids)
    
    ## print PCA plot    
    pca2D = decomposition.PCA(2)

    plot_columns = pca2D.fit_transform(df)

    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("k="+str(k_i))
    plt.show()

## Calculate the Silhouette score for Kmeans
def Kmeans_Silhouette(df,k_i):
    # Create clusters (If you try 3 and then 20, you will see how different
    # it looks when you attempt to fit the data.
    k = k_i
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(df)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    
Kmeans(normalizedDataFrame,5)
Kmeans(normalizedDataFrame,10)
Kmeans(normalizedDataFrame,2)
#Get the Silhouette for k=5,10,2
Kmeans_Silhouette(normalizedDataFrame,5)
Kmeans_Silhouette(normalizedDataFrame,10)
Kmeans_Silhouette(normalizedDataFrame,2)
    
## Function for Ward cluster  
def ward(df,k):
    n_clusters = k  # number of regions
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    ward.fit(df)
    label = np.reshape(ward.labels_, newshape=5465)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)
    
    ## print PCA plot    
    pca2D = decomposition.PCA(2)

    plot_columns = pca2D.fit_transform(df)

    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=label)
    plt.title("n="+str(k))
    plt.show()

##   Calculate the Silhouette score for Ward
def Ward_Silhouette(df,k):
    n_clusters = k  # number of regions
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    ward.fit(df)
    label = np.reshape(ward.labels_, newshape=5465)
        
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(df, label)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

    
ward(normalizedDataFrame,10)
ward(normalizedDataFrame,2)
Ward_Silhouette(normalizedDataFrame,5)
Ward_Silhouette(normalizedDataFrame,10)
Ward_Silhouette(normalizedDataFrame,2)
        
## Function for Ward Dbscan  
def Dbscan(df):
    dbscan = DBSCAN(eps=0.2, min_samples=10).fit(df)
    label = dbscan.labels_
        
    ## print PCA plot    
    pca2D = decomposition.PCA(2)

    plot_columns = pca2D.fit_transform(df)

    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=label)
    plt.title("Dbscan")
    plt.show() 
    
## Calculate the Silhouette score for DBscan
def Dbscan_Silhouette(df):
    dbscan = DBSCAN(eps=0.2, min_samples=10).fit(df)
    label = dbscan.labels_
        
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(df, label)
    print("The average silhouette_score is :", silhouette_avg)
    
Dbscan(normalizedDataFrame)    
Dbscan_Silhouette(normalizedDataFrame)    

## Asscociation rules
## Apriori
Aprioridf = mydf[mydf.columns[-6:]]
transactions  = Aprioridf.as_matrix()
## all results
results = list(apriori(transactions))

## Set Three different min support levels and print the itemset
for item in results:
    #print(item.items)
    if((item.support>0.15) & (item.support<0.2)):
        print(item)
        
for item in results:
    #print(item.items)
    if((item.support>0.125) & (item.support<0.15)):
        print(item)

for item in results:
    #print(item.items)
    if((item.support>0.10) & (item.support<0.125)):
        print(item)
    

for i in range(len(mydf["timestamp"])):
    temp = mydf.loc[i,"time"]
    mydf.loc[i,"day"]= temp[:10]
    
## Inputting the Twitter sentiment scores and add the to the data set
twitter = pd.read_csv('Twitter_sentiment_scores.csv' , sep=',', encoding='latin1')

for i in range(len(mydf['timestamp'])):
    temp = mydf.loc[i,"time"]
    day = str(temp[:10])
    for j in range(len(twitter['date'])):
        if day == twitter.loc[j,'date']:
            mydf.loc[i,'sentiment_labels']=twitter.loc[j,'sentiment_labels']
            mydf.loc[i,'sentiment_change']=twitter.loc[j,'sentiment_change']

mydf = mydf.dropna()

## Creat a data frame by days
Mean_by_day = pd.DataFrame(mydf.groupby(["day"]).mean())
del Mean_by_day['price']
del Mean_by_day['open']
del Mean_by_day['close']
del Mean_by_day['high']
del Mean_by_day['low']
del Mean_by_day['vwap']

## label the date
## 0:before the Announcement
## 1:After the Announcement
for  i in range(len(Mean_by_day["timestamp"])):
    if(Mean_by_day.ix[i,"timestamp"]<1505318412):
        Mean_by_day.ix[i,"category"] = 0
    else:
        Mean_by_day.ix[i,"category"] = 1
        
##reset the index
mydf_quant = mydf_quant.reset_index()
mydf=mydf.reset_index()
       
## label the date
for  i in range(len(mydf["timestamp"])):
    if(mydf.loc[i,"timestamp"]<1505289600):
        mydf.loc[i,"category"] = 0
    else:
        mydf.loc[i,"category"] = 1

##Multivariable Regression

from sklearn import linear_model
x = Mean_by_day[['category', 'Volatility']]  
y =np.array(Mean_by_day['Price Change'])

regr = linear_model.LinearRegression()
regr.fit(x,y.reshape(-1,1))

a, b = regr.coef_, regr.intercept_
print("Price Change=",a[0][0],"category+",a[0][1],"Volatility+",b[0])



## label the data, 0 if the price change is negative and 1 is positive
for i in range(len(mydf['Price Change'])):
    if mydf.loc[i,'Price Change'] <= 0:
        mydf['label']=0
    else:
        mydf['label']=1

## ROC curve and a confusion matrix for Decision Tree
def cm_roc(a):
    print("accuracy=", accuracy_score(Y_validate, predictions_2))
    print(confusion_matrix(Y_validate, predictions_2))
    cm_2 = confusion_matrix(Y_validate, predictions_2)
    classification_accuracy_2 = float((cm_2[0][0] + cm_2[1][1]))/float((cm_2[0][0] + cm_2[0][1] + cm_2[1][0] + cm_2[1][1])) #(a+d)/(a+b+c+d)
    print("classification_accuracy: " + str(classification_accuracy_2))
    print(classification_report(Y_validate, predictions_2))
    fpr, tpr, thres = roc_curve(Y_validate, predictions_2)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

## Classification Methods for data with twitter coefficient
## Produce a pure quantitative dataframe
mydf_quant = mydf.copy()
del mydf_quant['ticker']
del mydf_quant['time']
del mydf_quant['AAPL_Beta_Binning']
del mydf_quant['Volume_Binning']
del mydf_quant['timestamp']
del mydf_quant['Nasdaq_Beta_Binning']
del mydf_quant['Market_Volume_Binning']
del mydf_quant['Volatility_Binning']
del mydf_quant['Market_Volatility_Binning']
del mydf_quant['price']
del mydf_quant['open']
del mydf_quant['close']
del mydf_quant['high']
del mydf_quant['low']
del mydf_quant['vwap']
del mydf_quant['Price Change']
del mydf_quant['day']
del mydf_quant['index']
del mydf_quant['Nasdaq_price'] 
del mydf_quant['sentiment_labels']

##Create the list for models
models = []
models.append(('K-Neighbor', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Gaussian', GaussianNB()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('SVM', SVC(C=1, kernel="linear")))

##Normalize the data
x = mydf_quant.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalizedDataFrame = pd.DataFrame(x_scaled)

valueArray =  normalizedDataFrame.values
X = valueArray[:,0:5]
Y = valueArray[:,6]
for i in range(len(Y)):
    Y[i] = str(Y[i])

test_size = 0.20
seed = 7
X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

# Evaluate each model, add results to a results array,
# Print the accuracy results (remember these are averages and std)
results = []
names = []

print("Accuracy of Training Set")
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

print()
print("Accuracy of Validation Set")
for name, model in models:
    method = model
    method.fit(X_train, Y_train)
    predictions = method.predict(X_validate)
    print(name, accuracy_score(Y_validate, predictions))

## ROC and confusion matrix for each method
print("Decision Tree")
tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
predictions_2 = tree.predict(X_validate)

cm_roc(predictions_2)

print("Naive Bayes")
NB = GaussianNB()
NB.fit(X_train, Y_train)
predictions_2 = NB.predict(X_validate)

cm_roc(predictions_2)

print("Random Forest")
RF=RandomForestClassifier()
RF.fit(X_train, Y_train)
predictions_2 = RF.predict(X_validate)

cm_roc(predictions_2)

print("K-Neighbor")
KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)
predictions_2 = KNN.predict(X_validate)

cm_roc(predictions_2)


## Classification Methods for data without twitter coefficient
del mydf_quant['sentiment_change']

models = []
models.append(('K-Neighbor', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Gaussian', GaussianNB()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('SVM', SVC(C=1, kernel="linear")))


results = []
names = []

x = mydf_quant.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalizedDataFrame = pd.DataFrame(x_scaled)

valueArray =  normalizedDataFrame.values
X = valueArray[:,0:4]
Y = valueArray[:,5]
for i in range(len(Y)):
    Y[i] = str(Y[i])

test_size = 0.20
seed = 7
X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'


print("Accuracy of Training Set")
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

print()
print("Accuracy of Validation Set")
for name, model in models:
    method = model
    method.fit(X_train, Y_train)
    predictions = method.predict(X_validate)
    print(name, accuracy_score(Y_validate, predictions))


## ROC curve and a confusion matrix for Decision Tree
print("Decision Tree")
tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
predictions_2 = tree.predict(X_validate)

cm_roc(predictions_2)

print("Naive Bayes")
NB = GaussianNB()
NB.fit(X_train, Y_train)
predictions_2 = NB.predict(X_validate)

cm_roc(predictions_2)

print("SVM")
SVM = SVC(C=1, kernel="linear")
SVM .fit(X_train, Y_train)
predictions_2 = SVM.predict(X_validate)

cm_roc(predictions_2)

print("Random Forest")
RF=RandomForestClassifier()
RF.fit(X_train, Y_train)
predictions_2 = RF.predict(X_validate)

cm_roc(predictions_2)

print("K-Neighbor")
KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)
predictions_2 = KNN.predict(X_validate)

cm_roc(predictions_2)












