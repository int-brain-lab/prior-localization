import numpy as np
import pandas as pd
import decoding_utils as dut
import brainbox.io.one as bbone
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from brainbox.singlecell import calculate_peths
from sklearn.linear_model import Lasso

one = ONE()

# %% Run param definitions

SESS_CRITERION = 'aligned-behavior'
TARGET = 'signcont'
MODEL = expSmoothing_prevAction
MODELFIT_PATH = '/home/berk/Documents/Projects/prior-localization/results/inference/'
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (0, 0.1)
ESTIMATOR = Lasso

# %% Check if fits have been run on a per-subject basis
sessdf = dut.query_sessions(selection=SESS_CRITERION).sort_values('eid').set_index('eid')

for eid in np.unique(sessdf.index):
    subject = sessdf.loc[eid].iloc[0].subject
    tvec = dut.compute_target(TARGET, subject, eid)
    trialsdf = bbone.load_trials_df(eid, one=one)
    for subject, probe in sessdf.loc[eid]:
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                    one=one,
                                                                    probe=probe,
                                                                    aligned=True)
        beryl_reg = dut.remap_region(clusters[probe].atlas_id)
        regions = np.unique(beryl_reg)
        for region in regions:
            reg_clu = np.argwhere(beryl_reg == region).flatten()
            _, binned = calculate_peths(spikes[probe].times, spikes[probe].clusters, reg_clu,
                                        trialsdf[ALIGN_TIME], pre_time=TIME_WINDOW[0],
                                        post_time=TIME_WINDOW[1],
                                        bin_size=TIME_WINDOW[1] - TIME_WINDOW[0], smoothing=0,
                                        return_fr=False)
            binned = binned.squeeze().T
            if len(binned.shape) > 2:
                raise ValueError('Multiple bins are being calculated per trial,'
                                 'may be due to floating point representation error.'
                                 'Check window.')
            msub_binned = binned - np.mean(binned, axis=1)
        

# %%
