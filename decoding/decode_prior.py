import numpy as np
import pandas as pd
import decoding_utils as dut
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction

one = ONE()

# %% Run param definitions

SESS_CRITERION = 'aligned-behavior'
MODEL = expSmoothing_prevAction
MODELFIT_PATH = '/home/berk/Documents/Projects/prior-localization/results/inference/'

# %% Check if fits have been run on a per-subject basis
eids, probes, subjects = dut.query_sessions(selection=SESS_CRITERION, return_subjects=True)

for subject in np.unique(subjects):
    subjeids = eids[subjects == subject]
    dut.check_bhv_fit_exists(subject, MODEL, subjeids, MODELFIT_PATH))
# %%
