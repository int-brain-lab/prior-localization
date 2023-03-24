#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:15:07 2023

@author: bensonb
"""
# why are there more choice clusters in the old version?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from one.remote import aws
from one.webclient import AlyxClient
from one.api import ONE
from brainwidemap import download_aggregate_tables

# one = ONE()
# df = download_aggregate_tables(one, type='clusters')
ac = AlyxClient()

array_new = pd.read_csv('decoding_processing/clusters_regions_sessions/09-03-2023_choice_clusters.csv')
set_new = set(array_new.loc[:,array_new.columns[1]])

array_old = pd.read_csv('decoding_processing/18-01-2023_choice_clusteruuids_weights.csv')
set_old = set(array_old.loc[:,array_old.columns[1]])

new_minus_old = set_new.difference(set_old)
old_minus_new = set_old.difference(set_new)



# Download clusters table
src = 'aggregates/clusters.pqt'
dst = ac.cache_dir.parent / src
dst.parent.mkdir(exist_ok=True)
s3, bucket = aws.get_s3_from_alyx(ac)
file = aws.s3_download_file(src, dst, s3=s3, bucket_name=bucket)
# Load
df = pd.read_parquet(file).set_index('uuids')
master_uuids = set(df.index)

# get michael's cluster
out = pd.read_csv('/home/bensonb/Downloads/choice_weights.csv')
michael_uuids = set(out.loc[:, 'uuids'])

# Get pid and eid from cluster uuid
cluster_uuid = '84eacd02-02a9-4da5-b5b1-81ab26c8a40e'
cus = list(old_minus_new.intersection(master_uuids))
out = df.loc[cus, :]
out['eid_reg'] = out.apply(lambda x: f"{x['eid']}_{x['acronym']}", axis=1)
ers = np.unique(out.loc[:,'eid_reg'])