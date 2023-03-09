#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:42:36 2023

@author: bensonb
"""
import numpy as np
import pandas as pd

DATE = '18-01-2023'
VARI = 'block'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
block_cuuids = list(out['cluster_uuids'])

DATE = '18-01-2023'
VARI = 'stim'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
stim_cuuids = list(out['cluster_uuids'])

DATE = '18-01-2023'
VARI = 'choice'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
choice_cuuids = list(out['cluster_uuids'])

DATE = '18-01-2023'
VARI = 'feedback'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
feedback_cuuids = list(out['cluster_uuids'])

# DATE = '18-01-2023'
# VARI = 'wheel-speed'
# out = pd.read_csv(
#     f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
# wheel_cuuids = list(out['cluster_uuids'])

intersect_cuuids = list(set(block_cuuids).intersection(set(stim_cuuids)).intersection(set(choice_cuuids)).intersection(set(feedback_cuuids)))

pd.DataFrame(intersect_cuuids, 
             columns=['cluster_uuid']).to_csv('BWM_cluster_uuids.csv', 
                                              index=False)