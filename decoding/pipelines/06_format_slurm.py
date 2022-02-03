import pandas as pd
import pickle

indexers = ['subject', 'eid', 'probe', 'region']
indexers_neurometric = ['low_slope', 'high_slope', 'low_range', 'high_range', 'shift', 'mean_range', 'mean_slope']
resultslist = []
for fn in finished:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    for i_run in range(len(result['fit'])):
        tmpdict = {**{x: result[x] for x in indexers},
                   'fold': -1,
                   'pseudo_id': result['pseudo_id'],
                   'N_units': result['N_units'],
                   'run_id': i_run + 1,
                   'mask': ''.join([str(item) for item in list(result['fit'][i_run]['mask'].values * 1)]),
                   'R2_test': result['fit'][i_run]['Rsquared_test_full']}
        if result['fit'][i_run]['full_neurometric'] is not None:
            tmpdict = {**tmpdict,
                       **{idx_neuro: result['fit'][i_run]['full_neurometric'][idx_neuro]
                          for idx_neuro in indexers_neurometric}}
        resultslist.append(tmpdict)
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
                                                                                                        'prederr'] else 'task',
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