# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:52:37 2017

@author: Tianjing Cai, Hong Xu, Wen Cui, Jie He

This part is scrape stock data from Tradier Api. Get detailed Apple stock performance from 
Aug 28,2017 to Oct 05 every 5min and Sep 21&22 every 1min (as new iphone is released on Sep21).
Scrape the Nasdaq ETF(QQQ) at corresponding time. Output a dataframe containing information
of both Apple and Nasdaq.    

"""

import http.client
import pandas as pd
import json
import numpy as np
# Request: Market Quotes (https://sandbox.tradier.com/v1/markets/quotes?symbols=spy)

connection = http.client.HTTPSConnection('sandbox.tradier.com', 443, timeout = 30)

# Headers
Ticker_A = "AAPL"
StockName = "APPLE"
headers = {"Accept":"application/json",
           "Authorization":"Bearer AMpwYohTZAdVG2Fa8fEsBeuSsjUO"}

# Send synchronously
df = pd.DataFrame(columns = ['ticker','time','timestamp','price','open','high','low','close','volume','vwap'])

date_pair = [("2017-08-28", "2017-09-01"),
             ("2017-09-04", "2017-09-08"),("2017-09-11", "2017-09-15"),("2017-09-18", "2017-09-20")]
           
for dates in date_pair:
    connection.request('GET', '/v1/markets/timesales?symbol=AAPL&interval=5min&start='+dates[0]+'T07:30'+'&end='+dates[1]+'T19:55', None, headers)
    
    response = connection.getresponse()
    
    content = response.read().decode('utf-8')
    jsontxt = json.loads(content)
    #print(jsontxt)
    
    
    temp_dict = jsontxt['series']
    dict_store = temp_dict['data']
    for dictionary in dict_store:
        ticker = Ticker_A
        time = dictionary['time']
        timestamp = dictionary['timestamp']
        price = dictionary['price']
        
        open_price = dictionary['open']
        high_price = dictionary['high']
        low_price = dictionary['low']
        close_price = dictionary['close']
        volume = dictionary['volume']
        vwap = dictionary['vwap']
        df.loc[-1] = [ticker,time, timestamp, price, open_price, high_price, low_price, close_price, volume, vwap]
        df.index = df.index+1

# change data collection for sept 21 and sept 22 to 1 minute
connection.request('GET', '/v1/markets/timesales?symbol=AAPL&interval=1min&start=2017-09-21T07:30&end=2017-09-22T19:55', None, headers)
response = connection.getresponse()

content = response.read().decode('utf-8')
jsontxt = json.loads(content)
#print(jsontxt)


temp_dict = jsontxt['series']
dict_store = temp_dict['data']
for dictionary in dict_store:
    ticker = Ticker_A
    time = dictionary['time']
    timestamp = dictionary['timestamp']
    price = dictionary['price']
    
    open_price = dictionary['open']
    high_price = dictionary['high']
    low_price = dictionary['low']
    close_price = dictionary['close']
    volume = dictionary['volume']
    vwap = dictionary['vwap']
    df.loc[-1] = [ticker, time, timestamp, price, open_price, high_price, low_price, close_price, volume, vwap]
    df.index = df.index+1

date_pair = [("2017-09-25", "2017-09-29"),("2017-10-02", "2017-10-05")
             ]            
for dates in date_pair:
    connection.request('GET', '/v1/markets/timesales?symbol=AAPL&interval=5min&start='+dates[0]+'T07:30'+'&end='+dates[1]+'T19:55', None, headers)
    
    response = connection.getresponse()
    
    content = response.read().decode('utf-8')
    jsontxt = json.loads(content)
    #print(jsontxt)
    
    
    temp_dict = jsontxt['series']
    dict_store = temp_dict['data']
    for dictionary in dict_store:
        ticker = Ticker_A
        time = dictionary['time']
        timestamp = dictionary['timestamp']
        price = dictionary['price']
        
        open_price = dictionary['open']
        high_price = dictionary['high']
        low_price = dictionary['low']
        close_price = dictionary['close']
        volume = dictionary['volume']
        vwap = dictionary['vwap']
        df.loc[-1] = [ticker, time, timestamp, price, open_price, high_price, low_price, close_price, volume, vwap]
        df.index = df.index+1
df = df.sort_index()


## Now we collect the data for Nas
df2 = pd.DataFrame(columns = ['time','Nasdaq_price','Nasdaq_volume'])

date_pair = [("2017-08-28", "2017-09-01"),
             ("2017-09-04", "2017-09-08"),("2017-09-11", "2017-09-15"),("2017-09-18", "2017-09-20")]
           
for dates in date_pair:
    connection.request('GET', '/v1/markets/timesales?symbol=QQQ&interval=5min&start='+dates[0]+'T07:30'+'&end='+dates[1]+'T19:55', None, headers)
    
    response = connection.getresponse()
    
    content = response.read().decode('utf-8')
    jsontxt = json.loads(content)
    #print(jsontxt)
    
    
    temp_dict = jsontxt['series']
    dict_store = temp_dict['data']
    for dictionary in dict_store:
        time = dictionary['time']
        price = dictionary['price']
        volume = dictionary['volume']
        df2.loc[-1] = [time, price, volume]
        df2.index = df2.index+1

# change data collection for sept 21 and sept 22 to 1 minute
connection.request('GET', '/v1/markets/timesales?symbol=QQQ&interval=1min&start=2017-09-21T07:30&end=2017-09-22T19:55', None, headers)
response = connection.getresponse()

content = response.read().decode('utf-8')
jsontxt = json.loads(content)
#print(jsontxt)


temp_dict = jsontxt['series']
dict_store = temp_dict['data']
for dictionary in dict_store:
    time = dictionary['time']
    price = dictionary['price']
    volume = dictionary['volume']
    df2.loc[-1] = [time, price, volume]
    df2.index = df2.index+1

date_pair = [("2017-09-25", "2017-09-29"),("2017-10-02", "2017-10-05")
             ]            
for dates in date_pair:
    connection.request('GET', '/v1/markets/timesales?symbol=QQQ&interval=5min&start='+dates[0]+'T07:30'+'&end='+dates[1]+'T19:55', None, headers)
    
    response = connection.getresponse()
    
    content = response.read().decode('utf-8')
    jsontxt = json.loads(content)
    #print(jsontxt)
    
    
    temp_dict = jsontxt['series']
    dict_store = temp_dict['data']
    for dictionary in dict_store:
        time = dictionary['time']
        price = dictionary['price']
        volume = dictionary['volume']
        df2.loc[-1] = [time, price, volume]
        df2.index = df2.index+1
df2 = df2.sort_index()

##Merge two data frames by time
df3 = pd.DataFrame(columns = ['ticker','time','timestamp','price','open','high','low','close','volume','vwap','Nasdaq_price','Nasdaq_volume'])
df3 = pd.merge(df, df2, on='time', how='outer')

##Output the raw data
FileName = StockName+"_Stock_raw.csv"
with open(FileName, 'w') as output:
    df3.to_csv(output, sep = ',', index= False)
output.close()
  # Data scrape Success

  
"""
This part is data cleaning for the stock data. Modify the missing Nasdaq data by using preceding data 
and the Nasdaq Volume set as 0(just assume that there is no transaction). Besides, drop all rows with missing 
Apple data(useless). Finally, check and modify the invalid transaction for Apple stock.
"""

print("Read Data")
mydf = pd.read_csv('APPLE_Stock_raw.csv', sep=',', encoding='latin1')

## Quantitative Score for the cleanliness of current data frame
## Our  total noise and missing value
def cleanliness(df):
    mis_val = sum(mydf.isnull().sum())
    invalid = 0
    ##count rows with more than 100,000,000 or equal to 100
    for i in range(0,len(mydf['volume'])):
        if mydf.iloc[i]['volume'] > 10000000:
            invalid = invalid + 1
                    
        if mydf.iloc[i]['volume'] == 100:
            invalid = invalid + 1
    ##total noise and missing value
    total = invalid+mis_val
    cells = len(mydf['time'])*len(mydf.columns)
    score = total/cells
    print('The current cleanliness score is',1-score)
    print("Total noise:",invalid)
    print("Total missing:", mis_val)

cleanliness(mydf)
##Check the Amount of Nan and Duplicate Values
def Nan_table(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    return mis_val_table_ren_columns 

print("Total Missing Cells；",Nan_table(mydf)['Missing Values'].sum())
print(Nan_table(mydf))

## Check whether there is duplicate for time 
print("The number of rows with duplicated time:", sum(mydf.duplicated(['time', 'timestamp'])))


##For the Nasdaq ETF, if there are value is Nan, we assume price resumes unchanged and volume is 0
print("Cleaning Nasdaq Missing Value")
Nasdaq_changed = 0
for i in range(0,len(mydf['time'])):
    if np.isnan(mydf.iloc[i]['Nasdaq_price'])==True:
        mydf.loc[np.isnan(mydf['Nasdaq_volume'])==True, 'Nasdaq_volume'] = 0
        Nasdaq_changed = Nasdaq_changed+1
mydf['Nasdaq_price'].fillna(method='ffill', inplace=True)
print("The total rows of Nasdaq changed",Nasdaq_changed)
cleanliness(mydf)

##Delete the rows without Apple's data
print("Cleaning Apple Missing Value")
##print("Total Missing Cells(after filling Nasdaq missing values): ",Nan_table(mydf)['Missing Values'].sum())
##print(Nan_table(mydf))
mydf=mydf.dropna()
cleanliness(mydf)
print(Nan_table(mydf))

##For 5min interval data, we recognize any volume that larger than 100,000,000 should be invalid transaction
##And any volume which is exactly 100 means the transaction during the time is 0
print("Cleaning invalid transaction")
def check_invalid_transaction(mydf):
    invalid = 0
    ##drop rows with more than 100,000,000 transaction volume and turn 100 volume to 0
    for i in range(0,len(mydf['volume'])):
        if mydf.iloc[i]['volume'] > 10000000:
            invalid = invalid + 1
            ## Print the infomation of invalid transaction and drop the row
            print(mydf.iloc[i])
            mydf.drop([i])
        
        if mydf.iloc[i]['volume'] == 100:
            invalid = invalid + 1
    ##Turn the invalid transaction to 0
    mydf.loc[mydf['volume'] == 100, 'volume'] = 0
    return (invalid)
print("There are",check_invalid_transaction(mydf), 'invalid transaction(s)')
cleanliness(mydf)
print(Nan_table(mydf))
print("The data is cleaning now")

file = open("APPLE_Stock_Clean.csv","w") 
mydf.to_csv(file, sep=',', index=True)
file.close()
