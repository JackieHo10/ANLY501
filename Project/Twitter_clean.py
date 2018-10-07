#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:48:52 2017

@author: Tianjing Cai

This program read dirty twitter Data and fix incorrect values and remove duplicate rows
Also this program generate a cleanliness score that report how clean current data is after
implementing several cleaning methods and then output dataset to csv file
"""
import sys
import pandas as pd
import csv
import re
def main(argv):
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

if __name__ == "__main__":
    #execute only if run as a script
    main(sys.argv)