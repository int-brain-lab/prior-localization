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
LABELS = ['old_decoding_params','old_decoding', 'new']
DATES = ['24-03-2023', '25-03-2023', '26-03-2023']
VARI = 'choice'
VALUE = 'clusters' # 'clusters' or 'regions' or 'sessions'

date = DATES[0]
array1 = pd.read_csv(f'decoding_processing/clusters_regions_sessions/{date}_{VARI}_{VALUE}.csv')
date = DATES[1]
array2 = pd.read_csv(f'decoding_processing/clusters_regions_sessions/{date}_{VARI}_{VALUE}.csv')
date = DATES[2]
array3 = pd.read_csv(f'decoding_processing/clusters_regions_sessions/{date}_{VARI}_{VALUE}.csv')
# Define your sets
set1 = set(array1.loc[:,array1.columns[1]])
set2 = set(array2.loc[:,array2.columns[1]])
set3 = set(array3.loc[:,array3.columns[1]])

sets = list([set1, set2, set3])

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
for s1 in sets:
    row = []
    for s2 in sets:
        row.append(len(s1.intersection(s2)))
    intersection_matrix.append(row)

# Create the matrix plot
fig, ax = plt.subplots()
im = ax.imshow(intersection_matrix)

# Add axis labels
ax.set_xticks(range(len(sets)))
ax.set_yticks(range(len(sets)))
ax.set_xticklabels(LABELS)
ax.set_yticklabels(LABELS)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Add values to matrix
for i in range(len(sets)):
    for j in range(len(sets)):
        # text = ax.text(j, i, intersection_matrix[i][j],
        #                ha="center", va="center", color="w", fontsize=16)
        text = ax.text(j, i, intersection_matrix[i][j],
                       ha="center", va="center", color="g", fontsize=16)
plt.show()