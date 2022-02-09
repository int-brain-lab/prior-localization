import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
import utils
import pandas as pd
from one.api import ONE
from functions.neurometric import fit_get_shift_range
from functions.utils import optimal_Bayesian
import models.utils as mut
from scipy.stats import pearsonr, spearmanr, wilcoxon

# todo: try with difference of performances at 0 contrasts trials
one = ONE()
insdf = pd.read_parquet('neural/2022-02-05_decode_signcont_task_Lasso_align_goCue_times_100_pseudosessions_regionWise_timeWindow_-0_6_-0_1_neurometricPLeft_optimal_bayesian_pseudoSessions_unmergedProbes.parquet')
eids = insdf.index.get_level_values(1).unique()


outdict = {}
nb_simul_beh_shift = 10000

for eid in eids:
    print(eid)
    try:
        data = utils.load_session(eid, one=one)
    except:
        continue
    uniq_contrasts = np.array([-1., -0.25, -0.125, -0.0625, 0.,  0.0625,  0.125, 0.25,  1.])

    side, stim, act, _ = mut.format_data(data)

    pLeft_constrast = {c: np.mean(act[stim == c] == 1) for c in uniq_contrasts}
    no_integration_act = np.vstack([2 * np.random.binomial(1, pLeft_constrast[c], size=nb_simul_beh_shift) - 1 for
                                    c in stim])

    prior = optimal_Bayesian(act, stim, side).numpy()
    perfat0 = (act == side)[stim == 0].mean()
    t, p = wilcoxon((act == side)[stim == 0] * 1 - 0.5, alternative='greater')

    p_nointegration = np.mean((no_integration_act == side[:, None]).mean(axis=0) < (act == side).mean())

    p_0cont_nointegration = np.mean((no_integration_act[stim == 0] == side[stim == 0, None]).mean(axis=0)
                                    < (act == side)[stim == 0].mean())

    low_prob_idx_trials = [(prior < 0.5) * (stim == c) for c in uniq_contrasts]
    lowprob_arr = [uniq_contrasts,
                   [len(data['choice'][idx]) for idx in low_prob_idx_trials],
                   [(data['choice'][idx] == 1).mean() for idx in low_prob_idx_trials]]
    high_prob_idx_trials = [(prior > 0.5) * (stim == c) for c in uniq_contrasts]
    highprob_arr = [uniq_contrasts,
                    [len(data['choice'][idx]) for idx in high_prob_idx_trials],
                    [(data['choice'][idx] == 1).mean() for idx in high_prob_idx_trials]]

    full_neurometric = fit_get_shift_range([lowprob_arr, highprob_arr], False)

    outdict[eid] = [t, p, p_nointegration, p_0cont_nointegration, perfat0, full_neurometric['shift']]

behdf = pd.DataFrame.from_dict(outdict, orient='index', columns=['tvalue', 'pvalue', 'p_nointegration',
                                                                 'p_0cont_nointegration',
                                                                 '0contrast_perf', 'psychometric_shift'])
behdf.to_parquet('behavioral/beh_shift.parquet')

print(spearmanr(behdf['0contrast_perf'].values, behdf['psychometric_shift'].values))