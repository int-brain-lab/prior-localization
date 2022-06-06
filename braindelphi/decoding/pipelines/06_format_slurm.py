import pickle
from behavior_models.models.utils import format_data as format_data_mut
import pandas as pd
import glob
from braindelphi.decoding.settings import *
import models.utils as mut
from braindelphi.params import FIT_PATH
from braindelphi.decoding.settings import modeldispatcher


SAVE_KFOLDS = False

date = '06-05-2022'  # '06-05-2022' , month-day-year
finished = glob.glob(str(FIT_PATH.joinpath("*", "*", "*", "*%s*" % date)))

indexers = ['subject', 'eid', 'probe', 'region']
indexers_neurometric = ['low_slope', 'high_slope', 'low_range', 'high_range', 'shift', 'mean_range', 'mean_slope']
resultslist = []
from tqdm import tqdm
for fn in tqdm(finished):
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    if result['fit'] is None:
        continue
    for i_run in range(len(result['fit'])):
        side, stim, act, _ = format_data_mut(result["fit"][i_run]["df"])
        mask = result["fit"][i_run]["mask"]  # np.all(result["fit"][i_run]["target"] == stim[mask])
        full_test_prediction = np.zeros(np.array(result["fit"][i_run]["target"]).size)
        for k in range(len(result["fit"][i_run]["idxes_test"])):
            full_test_prediction[result["fit"][i_run]['idxes_test'][k]] = result["fit"][i_run]['predictions_test'][k]
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
                   'mask': ''.join([str(item) for item in list(result['fit'][i_run]['mask'].values * 1)]),
                   'R2_test': result['fit'][i_run]['Rsquared_test_full'],
                   'prediction': list(full_test_prediction),
                   # 'target': list(result["fit"][i_run]["target"]),
                   # 'perf_allcontrast': perf_allcontrasts,
                   # 'perf_allcontrasts_prevtrial': perf_allcontrasts_prevtrial,
                   # 'perf_0contrast': perf_0contrasts,
                   # 'nb_trials_act_is_0': nb_trials_act_is_0,
                   }
        if 'acc_test_full' in result['fit'][i_run].keys():
            tmpdict = {**tmpdict, 'acc_test': result['fit'][i_run]['acc_test_full'],
                       'balanced_acc_test': result['fit'][i_run]['balanced_acc_test_full']}
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
resultsdf = pd.DataFrame(resultslist)

estimatorstr = strlut[ESTIMATOR]
start_tw, end_tw = TIME_WINDOW
fn = str(FIT_PATH.joinpath('_'.join([date, 'decode', TARGET,
                                                               modeldispatcher[MODEL] if TARGET in ['prior',
                                                                                                        'pLeft']
                                                               else 'task',
                                                               estimatorstr, 'align', ALIGN_TIME, str(N_PSEUDO),
                                                               'pseudosessions',
                                                               'regionWise' if SINGLE_REGION else 'allProbes',
                                                               'timeWindow', str(start_tw).replace('.', '_'),
                                                               str(end_tw).replace('.', '_')])))
if COMPUTE_NEUROMETRIC:
    fn = fn + '_'.join(['', 'neurometricPLeft', modeldispatcher[MODEL]])

if ADD_TO_SAVING_PATH != '':
    fn = fn + '_' + ADD_TO_SAVING_PATH

fn = fn + '.parquet'

metadata_df = pd.Series({'filename': fn,  'date': date, **fit_metadata})
metadata_fn = '.'.join([fn.split('.')[0], 'metadata', 'pkl'])
resultsdf.to_parquet(fn)
metadata_df.to_pickle(metadata_fn)


"""
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
"""
