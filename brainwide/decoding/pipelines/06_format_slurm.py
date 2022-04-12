import pandas as pd
import pickle
import functions.utils as dut
import pandas as pd
import sys
import glob
import os
from settings.settings import *
import models.utils as mut

# import cached data
insdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt'))
insdf = insdf[insdf.spike_sorting != '']
eids = insdf['eid'].unique()

SAVE_KFOLDS = True

date = '2022-02-19'
finished = glob.glob(str(DECODING_PATH.joinpath("results", "neural", "*", "*", "*",
                                                "*%s*%s*%s*%s*%s*" % (date, TARGET, str(TIME_WINDOW[0]).replace('.', '_'),
                                                                      str(TIME_WINDOW[1]).replace('.', '_'),
                                                                      ADD_TO_SAVING_PATH))))

indexers = ['subject', 'eid', 'probe', 'region']
indexers_neurometric = ['low_slope', 'high_slope', 'low_range', 'high_range', 'shift', 'mean_range', 'mean_slope']
resultslist = []
trajectoriesdict = {}
for fn in finished:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    if result['fit'] is None:
        continue
    for i_run in range(len(result['fit'])):
        side, stim, act, _ = mut.format_data(result["fit"][i_run]["df"])
        mask = result["fit"][i_run]["mask"]  # np.all(result["fit"][i_run]["target"] == stim[mask])
        full_test_prediction = np.zeros(result["fit"][i_run]["target"].size)
        #for k in range(len(result["fit"][i_run]["idxes_test"])):
        #    full_test_prediction[result["fit"][i_run]['idxes_test'][k]] = result["fit"][i_run]['predictions_test'][k]
        #neural_act = np.sign(full_test_prediction)
        #perf_allcontrasts = (side.values[mask][neural_act != 0] == neural_act[neural_act != 0]).mean()
        #perf_allcontrasts_prevtrial = (side.values[mask][1:] == neural_act[:-1])[neural_act[:-1] != 0].mean()
        #perf_0contrasts = (side.values[mask] == neural_act)[(stim[mask] == 0) * (neural_act != 0)].mean()
        #nb_trials_act_is_0 = (neural_act == 0).mean()
        tmpdict = {**{x: result[x] for x in indexers},
                   'fold': -1,
                   'pseudo_id': result['pseudo_id'],
                   'N_units': result['N_units'],
                   'run_id': i_run + 1,
                   'prediction': list(full_test_prediction),
                   'target': list(result["fit"][i_run]["target"]),
                   #'perf_allcontrast': perf_allcontrasts,
                   #'perf_allcontrasts_prevtrial': perf_allcontrasts_prevtrial,
                   #'perf_0contrast': perf_0contrasts,
                   #'nb_trials_act_is_0': nb_trials_act_is_0,
                   'mask': ''.join([str(item) for item in list(result['fit'][i_run]['mask'].values * 1)]),
                   'R2_test': result['fit'][i_run]['Rsquared_test_full']}
        if result['fit'][i_run]['full_neurometric'] is not None:
            tmpdict = {**tmpdict,
                       **{idx_neuro: result['fit'][i_run]['full_neurometric'][idx_neuro]
                          for idx_neuro in indexers_neurometric}}
        resultslist.append(tmpdict)

        if SAVE_KFOLDS:
            for kfold in range(result['fit'][i_run]['nFolds']):
                tmpdict = {**{x: result[x] for x in indexers},
                           'fold': kfold,
                           'pseudo_id': result['pseudo_id'],
                           'N_units': result['N_units'],
                           'run_id': i_run + 1,
                           'R2_test': result['fit'][i_run]['Rsquareds_test'][kfold],
                           'Best_regulCoef': result['fit'][i_run]['best_params'][kfold],
                           }
                if result['fit'][i_run]['fold_neurometric'] is not None:
                    tmpdict = {**tmpdict,
                               **{idx_neuro: result['fit'][i_run]['fold_neurometric'][kfold][idx_neuro]
                                  for idx_neuro in indexers_neurometric}}
                resultslist.append(tmpdict)
resultsdf = pd.DataFrame(resultslist).set_index(indexers)

estimatorstr = strlut[ESTIMATOR]
start_tw, end_tw = TIME_WINDOW
fn = str(DECODING_PATH.joinpath('results', 'neural', '_'.join([DATE, 'decode', TARGET,
                                                               dut.modeldispatcher[MODEL] if TARGET in ['prior',
                                                                                                        'pLeft']
                                                               else 'task',
                                                               estimatorstr, 'align', ALIGN_TIME, str(N_PSEUDO),
                                                               'pseudosessions',
                                                               'regionWise' if SINGLE_REGION else 'allProbes',
                                                               'timeWindow', str(start_tw).replace('.', '_'),
                                                               str(end_tw).replace('.', '_')])))
if COMPUTE_NEUROMETRIC:
    fn = fn + '_'.join(['', 'neurometricPLeft', dut.modeldispatcher[MODEL]])

if ADD_TO_SAVING_PATH != '':
    fn = fn + '_' + ADD_TO_SAVING_PATH

fn = fn + '.parquet'

metadata_df = pd.Series({'filename': fn, **fit_metadata})
metadata_fn = '.'.join([fn.split('.')[0], 'metadata', 'pkl'])
resultsdf.to_parquet(fn)
metadata_df.to_pickle(metadata_fn)

with open(metadata_fn.split('.metadata.pkl')[0] + '.trajectories.pkl', 'wb') as f:
    pickle.dump(trajectoriesdict, f)

# save weights
weightsdict = {}
for fn in finished:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    for i_run in range(len(result['fit'])):
        weightsdict = {**weightsdict, **{(tuple(result[x] for x in indexers)
                                          + ('pseudo_id_{}'.format(result['pseudo_id']),
                                             'run_id_{}'.format(i_run + 1)))
                                         : np.vstack(result['fit'][i_run]['weights'])}}

with open(metadata_fn.split('.metadata.pkl')[0] + '.weights.pkl', 'wb') as f:
    pickle.dump(weightsdict, f)