# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:36:14 2017

@author: hongx
"""

import networkx as nx
import matplotlib.pyplot as plt
import community
import pandas as pd
import numpy as np

df = pd.read_csv('finance.csv')

newDF = pd.DataFrame()

price = np.array(df["price"])
time = np.array(df["day"])
index = np.array(df.index)

for i in range(len(time)):
    time[i]=time[i][5:10]

count = 0
for i in range(len(df["price"])):
    p = price[len(df["price"])-1-i]
    for j in range(index[len(df["price"])-1-i]-1):
        a = list()
        a.append(time[len(df["price"])-i-1])
        a.append(time[j])
        a.append(p-price[j])
        newDF[count]= a
        count = count+1

newDF = newDF.T

newDF.to_csv('edges.txt',sep=',',header=False, index=False)

FILE1=open('edges.txt', "rb")
G8=nx.read_edgelist(FILE1, delimiter=",",create_using=nx.Graph(), nodetype=str,data=[("weight", float)])
FILE1.close()
##print("G8 is:" ,G8.edges(data=True), "\n\n\n\n")
edge_labels = dict( ((u, v), d["weight"]) for u, v, d in G8.edges(data=True) )
pos = nx.random_layout(G8)
nx.draw(G8, pos, edge_labels=edge_labels, with_labels=True,edge_color="blue",width=0.5)
labels = nx.get_edge_attributes(G8,'weight')
##nx.draw_networkx_edge_labels(G8,pos,edge_labels=labels)
plt.show()

# Computer and print other stats    
nbr_nodes = nx.number_of_nodes(G8)
nbr_edges = nx.number_of_edges(G8)
nbr_components = nx.number_connected_components(G8)

print("Number of nodes:", nbr_nodes)
print("Number of edges:", nbr_edges)
print("Number of connected components:", nbr_components)

print(nx.density(G8))


