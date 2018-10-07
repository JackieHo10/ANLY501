#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:35:35 2017

@author: Tianjing Cai, Hong Xu, Wen Cui, Jie He
This program contains four parts: collecting and cleaning data of Twitter and stock market.
We also include the .py include only the Stock related code, Twitter scrapy and Twitter data cleaning file.
Please see the result by running each .py as the Twitter scrapy needs specially packages and the code in 
it is slightly modified. And please read Readme first.
"""


"""
This program use GotOldTweets package to find all twitters that sent past 7 days,
and we could use query to grab data online and we output all those dataset to csv file.
"""


import sys
import got3
import pandas as pd
import csv
import re
import numpy as np
import http.client
import json


# Scraped twitter using API package called "old_tweet"
def main(argv):
    print('open csv file: ')
    outputFileName = "Twitter_output_dirty.csv"
    
    output = open(outputFileName, 'w')
    writer = csv.writer(output)
    # write header to file
    writer.writerow(["username", "date", "retweets", "favorites", "text", "geo", "mentions", "hashtags", "id", "permalink"])
    output.close()
    
    
    output = open(outputFileName, 'a')
    writer = csv.writer(output)
    year = 2017
    # create month-date pair that we want to grab information at
    month_data_pair = {"08":list(range(28,31)), "09":list(range(1,30)), "10":list(range(1,5))}
    for month, date_store in month_data_pair.items():
        for date in date_store:
            Start_YEAR_MONTH_DATE = str(year)+"-"+month+"-"+str(date) # find tweeter sent in the date range 
            End_YEAR_MONTH_DATE = str(year)+"-"+month+"-"+str(date+1) 
            # set tweeter search criteria: english only, 1000 tweets maximum per day, and time range given
            tweetCriteria = got3.manager.TweetCriteria().setQuerySearch('APPLE').setLang('en').setSince(Start_YEAR_MONTH_DATE).setUntil(End_YEAR_MONTH_DATE).setMaxTweets(1000)
            # get information that store all tweets
            tweets = got3.manager.TweetManager.getTweets(tweetCriteria)
            
            for t in tweets:
                # write each tweet to csv file
                writer.writerow([t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink])
    output.close()
    
    Twitter_DF = pd.read_csv("Twitter_output_dirty.csv" , sep=',', encoding='latin1')
    
    # regular expression to match key words in text attribute
    regex1 = r'.*[aA][Pp][Pp][Ll][Ee].* | .*[Ii][Pp][Hh][Oo][Nn][Ee].*|.*[Ww][Aa][Tt][Cc]\
    [Hh].*|.*[Ii][Tt][Uu][Nn][Ee][Ss].*|.*[Ii][Pp][Aa][Dd].*|.*[Ii][Mm][Aa][Cc].*|.*[Pp][Oo][Dd].*|\
        .*[Aa][Rr].*|.*[Tt][Vv].*|.*Tim.*|.*Steve.*|.*event.*|.*announce.*|.*release.*' 
    
    # regular expression of all those non-alphabetic/number/space    
    regex2=re.compile('[^A-Za-z0-9\s\d\.]+') 
    
    # regular expression to find link in text attribute
    regex3= re.compile('https.+$ | http.+$|pic\.twitter\.com.*$') 
    id_link = {}
    irre_count = 0
    dirty_count = 0 # count number of that has dirty value in attribute "text"
    duplicate_count = 0
    format_wrong = 0
    dirty_index = [] # store index of duplicate rows
    print("Begin to find incorrect / missing values from dataframe: ")
    for i in Twitter_DF.index:
        person_link = Twitter_DF.ix[i, "permalink"]
        
        ID = Twitter_DF.ix[i, "id"]
        # check number of digits for each id
        if( not len(str(ID)) == 18):
            
            format_wrong = format_wrong +1
        
        text = Twitter_DF.ix[i, "text"]
        
        # remove link in text
        if bool(re.search(regex3, text)):
            dirty_count = dirty_count +1
            
            Twitter_DF.ix[i, "text"] = re.sub(regex3, "", str(Twitter_DF.ix[i, "text"])) 
        
        # clean text data: remove those non alphabetic symbol / digits/ space and dot
        if bool(re.search(regex2, text)):
            dirty_count = dirty_count +1
            
            Twitter_DF.ix[i, "text"] = re.sub(regex2, "", str(Twitter_DF.ix[i, "text"])) 
        
        # store row indices that does not contain relevant keywords
        if not bool(re.search(regex1,text)):
            irre_count = irre_count +1
            dirty_index.append(i)
        
        # store row indices that has duplicate twitter post
        if ID in id_link.keys() and id_link[ID] == Twitter_DF.ix[i, "permalink"]:
            duplicate_count = duplicate_count +1
            dirty_index.append(i)
        
        # store post id and its link 
        else:
            id_link[ID] = Twitter_DF.ix[i, "permalink"]
            
    final_clean_percent = len(Twitter_DF.index) - irre_count - dirty_count - duplicate_count - format_wrong
    first_clean_percent = len(Twitter_DF.index) - dirty_count - format_wrong
    second_clean_percent = len(Twitter_DF.index) - dirty_count - duplicate_count - format_wrong
    third_clean_pencent = len(Twitter_DF.index) - dirty_count - duplicate_count - irre_count - format_wrong
    
    print("After removing dirty value, dirty score", str(final_clean_percent/first_clean_percent))
    print("After removing number of duplicate rows, dirty score: ", str(final_clean_percent/second_clean_percent))
    print("After removing number of rows that has irrelavant information, dirty score:  ", str(final_clean_percent/third_clean_pencent))
    
    print("Begin drop duplicate rows: ")
    for index in dirty_index:
        
        Twitter_DF.drop(index)
        
   #word_list = nltk.Text(all_word)
    #word_list.findall(r"<[Ii][Pp][Hh][Oo][Nn][Ee]><is>(<.*>)") 
    print("output our dataframe: ")
    outputFileName = "Twitter_output_cleaned.csv"    
    with open(outputFileName, 'w') as output:  
        Twitter_DF.to_csv(output, sep = ',')
    output.close()
    
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


if __name__ == "__main__":
    #execute only if run as a script
    main(sys.argv)