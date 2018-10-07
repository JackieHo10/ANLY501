# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:48:25 2017

@author: hongx
"""

import pandas as pd
import numpy as np

import plotly
plotly.tools.set_credentials_file(username='hongxu1013041980', api_key='1VNSjUpLmemhbUKZLQ9q')
import plotly.plotly as py
from plotly.offline import plot
import pandas as pd
import plotly.graph_objs as go
import plotly

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
        
        
for i in range(len(mydf['Price Change'])):
    if mydf.loc[i,'Price Change'] <= 0:
        mydf.loc[i,'label']=0
    else:
        mydf.loc[i,'label']=1
        
                
        
y0 = mydf[mydf['label']==0]['Volatility']
y1 = mydf[mydf['label']==1]['Volatility']

trace0 = go.Box(
    y=y0,
    name = 'Decrease',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
    
trace1 = go.Box(
    y=y1,
    name = 'Increase',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)

myLayout = {
  "title": "Boxplot for Volatility Group Apple stock's Increase/Decrease",
  "xaxis": {
    "title": "Increase/Decrease"
  }, 
  "yaxis": {
    "title" : 'Volatility'
  }
}
data = [trace0, trace1]
figure = go.Figure(data = data,layout = myLayout)


## plot the boxplot
myFigure = plot(figure,filename = "112.html")


trace = go.Scatter(
	x = mydf['time'],
	y = mydf['price'],
   mode = 'line',
   name = 'AAPL'
)

# Assign it to an iterable object named myData
myData = [trace]

# Add axes and title
myLayout = go.Layout(
	title = "AAPL Price over Time",
	xaxis=dict(
		title = 'Time'
	),
	yaxis=dict(
		title = 'Price'
	)
)

# Setup figure
myFigure = go.Figure(data=myData, layout=myLayout)

# Create the scatterplot
py.iplot(myFigure, filename='AAPL Price')


for i in range(len(mydf["timestamp"])):
    temp = mydf.loc[i,"time"]
    mydf.loc[i,"day"]= temp[:10]

inc_count = []
dec_count = []
time_list = []

mydf = mydf.sort_values(['day'])
day= mydf.ix[0,"day"]
time_list.append(day)
inc = 0
dec = 0


for i in range(len(mydf["timestamp"])):
    ##print(mydf.ix[i,"day"])
    if mydf.ix[i,"day"] != day:
        day = mydf.ix[i,"day"]
        time_list.append(day)
        inc_count.append(inc)
        dec_count.append(dec)
        inc = 0
        dec = 0
    if mydf.ix[i,"Price Change"] <= 0:
        dec = dec+1
    else:
        inc = inc+1
        
for i in range(len(inc_count)):
    if inc_count[i]>300:
        inc_count[i] = int(inc_count[i]/5)
        dec_count[i] = int(dec_count[i]/5)
        

inc_count.append(inc)
dec_count.append(dec)

trace1 = go.Bar(
    x= time_list,
    y=inc_count,
    name='Increase'
)
trace2 = go.Bar(
    x=time_list,
    y=dec_count,
    name='Decrease'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title = "Count of Increase/Decrease for each day",
	xaxis=dict(
		title = 'Time'
	),
	yaxis=dict(
		title = 'Count for Increase\Decrease'
	)    
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


Mean_by_day = pd.DataFrame(mydf.groupby(["day"]).mean())

print(Mean_by_day.index.tolist())

time_list.sort()

trace1 = go.Bar(
    x= time_list,
    y= Mean_by_day["Volatility"],
    name='Volatility'
)

data = [trace1]

layout = go.Layout(
	title = "Mean Volatility of each Day",
	xaxis=dict(
		title = 'Time'
	),
	yaxis=dict(
		title = 'Volatility'
	)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar_Volatility')





















