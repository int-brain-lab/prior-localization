import os
import pickle
import logging
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import decoding_utils as dut
import models.utils as mut
import brainbox.io.one as bbone
import sklearn.linear_model as sklm
from pathlib import Path
from datetime import date
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from brainbox.singlecell import calculate_peths
from tqdm import tqdm

one = ONE()
logger = logging.getLogger('ibllib')
logger.disabled = True
# %% Run param definitions

SESS_CRITERION = 'aligned-behavior'
TARGET = 'signcont'
MODEL = expSmoothing_prevAction
MODELFIT_PATH = '/home/berk/Documents/Projects/prior-localization/results/inference/'
OUTPUT_PATH = '/home/berk/Documents/Projects/prior-localization/results/decoding/'
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.6, -0.2)
ESTIMATOR = sklm.Lasso
N_PSEUDO = 10 #200
DATE = str(date.today())
deterministic_pseudoSessions = True

HPARAM_GRID = {'alpha': np.array([0.001, 0.01, 0.1])}

# %% Define helper functions to make main loop readable
def save_region_results(fit_result, pseudo_results, subject, eid, probe, region):
    subjectfolder = Path(OUTPUT_PATH).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    for folder in [subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    fn = '_'.join([DATE, region]) + '.pkl'
    fw = open(probefolder.joinpath(fn), 'wb')
    outdict = {'fit': fit_result, 'pseudosessions': pseudo_results,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region}
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


# %% Check if fits have been run on a per-subject basis
sessdf = dut.query_sessions(selection=SESS_CRITERION)
sessdf = sessdf.sort_values('subject').set_index(['subject', 'eid'])

filenames = []

# select eid
eid_list = ['dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', '034e726f-b35f-41e0-8d6c-a22cc32391fb',
            '7939711b-8b4d-4251-b698-b97c1eaa846e', 'fa704052-147e-46f6-b190-a65b837e605e',
            '46794e05-3f6a-4d35-afb3-9165091a5a74', '1538493d-226a-46f7-b428-59ce5f43f0f9',
            'b52182e7-39f6-4914-9717-136db589706e', '89f0d6ff-69f4-45bc-b89e-72868abb042a',
            '3ce452b3-57b4-40c9-885d-1b814036e936', '2d5f6d81-38c4-4bdc-ac3c-302ea4d5f46e',
            '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', 'd839491f-55d8-4cbe-a298-7839208ba12b',
            'd2918f52-8280-43c0-924b-029b2317e62c', 'c99d53e6-c317-4c53-99ba-070b26673ac4',
            'ecb5520d-1358-434c-95ec-93687ecd1396', '5386aba9-9b97-4557-abcd-abc2da66b863',
            '4b00df29-3769-43be-bb40-128b1cba6d35', '3663d82b-f197-4e8b-b299-7b803a155b84',
            '85dc2ebd-8aaf-46b0-9284-a197aee8b16f', '15f742e1-1043-45c9-9504-f1e8a53c1744']

# eid_list = ['034e726f-b35f-41e0-8d6c-a22cc32391fb'] #['46794e05-3f6a-4d35-afb3-9165091a5a74']
# sessdf.index.unique(level='eid')[115] #sessdf.index.unique(level='eid')[0]
all_outputs = dict()

for eid in eid_list:
    eid_output = dict()

    subject = sessdf.xs(eid, level='eid').index[0]
    subjeids = sessdf.xs(subject, level='subject').index.unique()

    behavior_data = mut.load_session(eid, one=one)
    try:
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, beh_data=behavior_data, one=one)
    except ValueError:
        print('Model not fit.')
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, one=one)

    msub_tvec = tvec  # - np.mean(tvec)
    trialsdf = bbone.load_trials_df(eid, one=one)

    # select probe
    probe = sessdf.loc[subject, eid, :].probe[0] #'probe00' #
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                one=one,
                                                                probe=probe,
                                                                aligned=True)
    beryl_reg = dut.remap_region(clusters[probe].atlas_id)
    regions = np.unique(beryl_reg)

    # select all regions
    reg_clu = np.unique(spikes[probe].clusters)  # np.arange(len(beryl_reg)).flatten()
    assert (np.all(np.unique(spikes[probe].clusters) == reg_clu)), 'problem in region selection'

    _, binned = calculate_peths(spikes[probe].times, spikes[probe].clusters, reg_clu,
                                trialsdf[ALIGN_TIME] + TIME_WINDOW[0], pre_time=0,
                                post_time=TIME_WINDOW[1] - TIME_WINDOW[0],
                                bin_size=TIME_WINDOW[1] - TIME_WINDOW[0], smoothing=0,
                                return_fr=False)
    binned = binned.squeeze()
    if len(binned.shape) > 2:
        raise ValueError('Multiple bins are being calculated per trial,'
                         'may be due to floating point representation error.'
                         'Check window.')
    msub_binned = binned  # - np.mean(binned, axis=0)
    fit_result = dut.regress_target(msub_tvec, msub_binned, ESTIMATOR(),
                                    hyperparam_grid=HPARAM_GRID, verbose=False, shuffle=False)

    # computing neurometric curve
    from decoding_stimulus_neurometric_fit import get_target_df, fit_get_shift_range

    lowprob_arr, highprob_arr = get_target_df(fit_result['target'],
                                              fit_result['prediction'],
                                              fit_result['idxes_test'],
                                              trialsdf,
                                              one)

    params, low_slope, high_slope, low_range, high_range, shift = fit_get_shift_range(lowprob_arr,
                                                                                      highprob_arr,
                                                                                      seed_=0)
    print(' low_slope:{} \n high_slope:{} \n low_range:{} \n high_range:{} \n shift:{}'. \
          format(low_slope, high_slope, low_range, high_range, shift))

    # save results
    for s in ['Rsquared_train', 'Rsquared_test', 'weights', 'target', 'prediction', 'idxes_test']:
        eid_output[s] = fit_result[s]
    eid_output['probe'] = probe
    eid_output['neurometric'] = {"low_slope": low_slope,
                                 "high_slope": high_slope,
                                 "low_range": low_range,
                                 "high_range": high_range,
                                 "mean_range": (low_range + high_range) / 2.,
                                 "shift": shift}

    from brainbox.task.closed_loop import generate_pseudo_session

    # pseudo sessions
    pseudo_results = dict()
    for pseudosess_idx, _ in enumerate(tqdm(range(N_PSEUDO), desc='Pseudosess: ', leave=False)):
        result_dict = dict()

        np.random.seed(pseudosess_idx)
        pseudosess = generate_pseudo_session(trialsdf)

        pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                         MODELFIT_PATH,
                                         modeltype=MODEL, beh_data=pseudosess, one=one)

        # msub_pseudo_tvec = pseudo_tvec - np.mean(pseudo_tvec)
        pseudo_result = dut.regress_target(pseudo_tvec, msub_binned, ESTIMATOR(),
                                           hyperparam_grid=HPARAM_GRID, shuffle=False)

        # get neurometric curves
        pseudo_lowprob_arr, pseudo_highprob_arr = get_target_df(pseudo_result['target'],
                                                                pseudo_result['prediction'],
                                                                pseudo_result['idxes_test'],
                                                                pseudosess,
                                                                one)

        params, low_slope, high_slope, low_range, high_range, shift = fit_get_shift_range(pseudo_lowprob_arr,
                                                                                          pseudo_highprob_arr,
                                                                                          seed_=0)

        for s in ['Rsquared_train', 'Rsquared_test', 'weights', 'target', 'prediction', 'idxes_test']:
            result_dict[s] = pseudo_result[s]
        result_dict['neurometric'] = {"low_slope": low_slope,
                                      "high_slope": high_slope,
                                      "low_range": low_range,
                                      "high_range": high_range,
                                      "mean_range": (low_range + high_range)/2.,
                                      "shift": shift}
        pseudo_results['{}'.format(pseudosess_idx)] = result_dict

    eid_output['pseudo_results'] = pseudo_results

    all_outputs[eid] = eid_output

# plot of range over the list of sessions
eids_mean_ranges = np.array([all_outputs[e]['neurometric']['mean_range'] for e in eid_list])
eids_shifts = np.array([all_outputs[e]['neurometric']['shift'] for e in eid_list])
eids_probes = np.array([all_outputs[e]['probe'] for e in eid_list])
eids_target = np.array([all_outputs[e]['target'] for e in eid_list])
eids_probes = np.array([all_outputs[e]['binned'] for e in eid_list])

pseudosess_mean_ranges = np.array([[all_outputs[e]['pseudo_results'][str(i)]['neurometric']['mean_range']
                                    for i in range(N_PSEUDO)] for e in eid_list])
pseudosess_shifts = np.array([[all_outputs[e]['pseudo_results'][str(i)]['neurometric']['shift']
                                    for i in range(N_PSEUDO)] for e in eid_list])

lite = {'eid_list':np.array(eid_list),
        'eids_mean_ranges':eids_mean_ranges,
        'eids_shifts':eids_shifts,
        'eids_probes':eids_probes,
        'pseudosess_mean_ranges':pseudosess_mean_ranges,
        'pseudosess_shifts':pseudosess_shifts}
pickle.dump(lite, open('sanity_analysis/summary_20sessions.pkl', 'wb'))

plt.figure()
plt.plot(pseudosess_mean_ranges, pseudosess_shifts, 'o', color='orange', label='pseudosession')
plt.plot(eids_mean_ranges, eids_shifts, 'o', color='blue', label='session')
plt.ylabel('shift')
plt.xlabel('range')
plt.legend()
plt.gca().set_xlim(0, 1)
plt.gca().set_ylim(-0.6, 0.6)
plt.show()
plt.savefig('figures/20sessions_sanitytest.pdf')

# tailored code to save particular informations of one session
Rsquares_pseudosession = dict()
Rsquares_pseudosession['Rsquared_trains'] = np.array([l['Rsquared_train'] for l in pseudo_results])
Rsquares_pseudosession['Rsquared_tests'] = np.array([l['Rsquared_test'] for l in pseudo_results])
Rsquares_pseudosession['target'] = np.array([l['target'] for l in pseudo_results])
Rsquares_pseudosession['test_indexes'] = np.array([l['idxes_test'] for l in pseudo_results])
import pickle
pickle.dump(Rsquares_pseudosession, open('Rsquares_pseudosession_nonShuffled_quasiRandom_1.pkl', 'wb'))
Rsquares_pseudosession_0 = pickle.load(open('Rsquares_pseudosession_nonShuffled_quasiRandom.pkl', 'rb'))
Rsquares_pseudosession_1 = pickle.load(open('Rsquares_pseudosession_nonShuffled_quasiRandom_1.pkl', 'rb'))
assert ([np.all(Rsquares_pseudosession_0[k] == Rsquares_pseudosession_1[k]) for k in Rsquares_pseudosession_0.keys()])

# loading results and plotting R2 over pseudosessions for one session
R2_pseudosess_nonShuffled = pickle.load(open('Rsquares_pseudosession_nonShuffled.pkl', 'rb'))
R2_pseudosess_shuffled = pickle.load(open('Rsquares_pseudosession_shuffled.pkl', 'rb'))
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.hist(R2_pseudosess_shuffled['Rsquared_trains'], histtype=u'step', density=True, linewidth=2, label='shuffled')
plt.hist(R2_pseudosess_nonShuffled['Rsquared_trains'], histtype=u'step', density=True, linewidth=2, label='unshuffled')
plt.gca().set_xlim(-1, 1)
plt.ylabel('density')
plt.xlabel('R2')
plt.title('train set', fontsize=15)
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(R2_pseudosess_shuffled['Rsquared_tests'], histtype=u'step', density=True, linewidth=2, label='shuffled')
plt.hist(R2_pseudosess_nonShuffled['Rsquared_tests'], histtype=u'step', density=True, linewidth=2, label='unshuffled')
plt.gca().set_xlim(-1, 1)
plt.ylabel('density')
plt.xlabel('R2')
plt.title('test set', fontsize=15)
plt.legend()
plt.suptitle('R2 on pseudo sessions: shuffled vs unshuffled train/test definition', fontsize=15)
# plt.tight_layout(pad=1.0)
plt.savefig('figures/R2_pseudosession_shuffled_unshuffed.pdf')
