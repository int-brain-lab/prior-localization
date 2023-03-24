#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:28:17 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


############## compare variables of newest run
DATE = '09-03-2023'
VARIS = ['block', 'choice', 'stimside']
VALUE = 'regions' # 'clusters' or 'regions' or 'sessions'

vari = VARIS[0]
array1 = pd.read_csv(f'decoding_processing/clusters_regions_sessions/{DATE}_{vari}_{VALUE}.csv')
vari = VARIS[1]
array2 = pd.read_csv(f'decoding_processing/clusters_regions_sessions/{DATE}_{vari}_{VALUE}.csv')
vari = VARIS[2]
array3 = pd.read_csv(f'decoding_processing/clusters_regions_sessions/{DATE}_{vari}_{VALUE}.csv')
# Define your sets
set1 = set(array1.loc[:,array1.columns[1]])
set2 = set(array2.loc[:,array2.columns[1]])
set3 = set(array3.loc[:,array3.columns[1]])

################### compare newest run to old run

# DATES = ['18-01-2023', '18-01-2023', '18-01-2023']
# VARIS = ['block', 'choice', 'stim']

# date = DATES[0]
# vari = VARIS[0]
# array1 = pd.read_csv(f'decoding_processing/{date}_{vari}_clusteruuids_weights.csv')
# set1 = set(array1.loc[:,array1.columns[1]])
# date = DATES[1]
# vari = VARIS[1]
# array2 = pd.read_csv(f'decoding_processing/{date}_{vari}_clusteruuids_weights.csv')
# set2 = set(array2.loc[:,array2.columns[1]])
# date = DATES[2]
# vari = VARIS[2]
# array3 = pd.read_csv(f'decoding_processing/{date}_{vari}_clusteruuids_weights.csv')
# set3 = set(array3.loc[:,array3.columns[1]])

# Create a matrix of pairwise intersections
intersection_matrix = []
for s1 in [set1, set2, set3]:
    row = []
    for s2 in [set1, set2, set3]:
        row.append(len(s1.intersection(s2)))
    intersection_matrix.append(row)

# Create the matrix plot
fig, ax = plt.subplots()
im = ax.imshow(intersection_matrix)

# Add axis labels
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(VARIS)
ax.set_yticklabels(VARIS)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Add values to matrix
for i in range(3):
    for j in range(3):
        # text = ax.text(j, i, intersection_matrix[i][j],
        #                ha="center", va="center", color="w", fontsize=16)
        text = ax.text(j, i, intersection_matrix[i][j],
                       ha="center", va="center", color="g", fontsize=16)
plt.show()