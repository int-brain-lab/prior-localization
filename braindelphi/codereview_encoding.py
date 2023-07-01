#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:46:09 2023

@author: bensonb
"""
from sklearn.linear_model import Ridge
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = Ridge(alpha=1.0)
clf.fit(X, y)
print(clf.coef_[0], clf.intercept_)
clf2 = Ridge(alpha=1000000.0)
clf2.fit(X, y)
print(clf2.coef_[0], clf2.intercept_)
print(np.mean(y))

#%%
import pandas as pd

VARI = 'wheel-speed'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_allProbes_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

# load results
res_table = pd.read_csv(file_all_results)
res_table.loc[:,0]

#%%

import pandas as pd
import numpy as np

out = pd.DataFrame(index=np.arange(100), columns=[1,2,3])
out.loc[:,:] = False

masks_set = out.groupby([1,2,3]).groups
for current_mask in masks_set:
    print(masks_set[current_mask])    
    
#%%

import numpy as np
np.ix_([1,2,3],[1,2,3])

#%%

import pandas as pd

vs = ['cluster1', 'cluster2', 'cluster3']
cs = np.arange(1)
mindex = pd.MultiIndex.from_product([vs, cs],
                                    names=["clu_id", "feature_iter"])

cols = ['stimR', 'stimL', 'wheel', 'corrFeedback']
out = pd.DataFrame(index=mindex, columns=cols)
out.loc[:,:] = np.random.rand(len(vs)*len(cs), len(cols))

print(out)
print()

for k in out.columns:
    print(1.0 - out[k].loc[:,0])
    print()

#%%

import pandas as pd

vs = ['cluster1', 'cluster2', 'cluster3']
cs = np.arange(2)
mindex = pd.MultiIndex.from_product([vs, cs],
                                    names=["clu_id", "feature_iter"])

cols = ['stimR', 'stimL', 'wheel', 'corrFeedback']
out = pd.DataFrame(index=mindex, columns=cols)
out.loc[:,:] = np.random.rand(len(vs)*len(cs), len(cols))

print(out)
print(out.groupby(['clu_id','feature_iter']).agg({'stimR':'mean','stimL':'mean'}))
print(out.groupby(['feature_iter']).agg({'stimR':'mean','stimL':'mean'}))
