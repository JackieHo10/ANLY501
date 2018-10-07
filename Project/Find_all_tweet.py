#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:35:35 2017

@author: Tianjing Cai
This program use GotOldTweets package to find all twitters that sent past 7 days,
and we could use query to grab data online and we output all those dataset to csv file.
"""


import sys
import got3
import pandas as pd
import csv
import re
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
    print('Begin to gather data from twitter and write to csv file: ')
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

if __name__ == "__main__":
    #execute only if run as a script
    main(sys.argv)