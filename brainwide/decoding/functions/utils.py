import os
import numpy as np
import pandas as pd
import models.utils as mut
from pathlib import Path
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from models.expSmoothing_stimside import expSmoothing_stimside
from models.biasedApproxBayesian import biased_ApproxBayesian
from models.biasedBayesian import biased_Bayesian
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model._coordinate_descent import LinearModelCV
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import torch
import pickle
import one.alf.io as alfio
import openturns
from brainbox.task.closed_loop import generate_pseudo_blocks, _draw_position, _draw_contrast


def query_sessions(selection='all', one=None):
    '''
    Filters sessions on some canonical filters
    returns dataframe with index being EID, so indexing results in subject name and probe
    identities in that EID.
    '''
    one = one or ONE()
    if selection == 'all':
        # Query all ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                                   'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all sessions with at least one alignment
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                                   'session__qc__lt,50,'
                                   'json__extended_qc__alignment_count__gt,0')
    elif selection == 'resolved':
        # Query all sessions with resolved alignment
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                                   'session__qc__lt,50,'
                                   'json__extended_qc__alignment_resolved,True')
    elif selection == 'aligned-behavior':
        # Query sessions with at least one alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                                   'session__qc__lt,50,'
                                   'json__extended_qc__alignment_count__gt,0,'
                                   'session__extended_qc__behavior,1')
    elif selection == 'resolved-behavior':
        # Query sessions with resolved alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                                   'session__qc__lt,50,'
                                   'json__extended_qc__alignment_resolved,True,'
                                   'session__extended_qc__behavior,1')
    else:
        raise ValueError('Invalid selection was passed.'
                         'Must be in [\'all\', \'aligned\', \'resolved\', \'aligned-behavior\','
                         ' \'resolved-behavior\']')

    #  Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    all_pids = np.array([i['id'] for i in ins])
    retdf = pd.DataFrame({'subject': all_subjects, 'eid': all_eids, 'probe': all_probes, 'pid': all_pids})
    retdf.sort_values('subject', inplace=True)
    return retdf


def get_target_pLeft(nb_trials, nb_sessions, take_out_unbiased, bin_size_kde):
    contrast_set = np.array([0., 0.0625, 0.125, 0.25, 1])
    target_pLeft = []
    for _ in np.arange(nb_sessions):
        pseudo_trials = pd.DataFrame()
        pseudo_trials['probabilityLeft'] = generate_pseudo_blocks(nb_trials)
        for i in range(pseudo_trials.shape[0]):
            position = _draw_position([-1, 1], pseudo_trials['probabilityLeft'][i])
            contrast = _draw_contrast(contrast_set, 'uniform')
            if position == -1:
                pseudo_trials.loc[i, 'contrastLeft'] = contrast
            elif position == 1:
                pseudo_trials.loc[i, 'contrastRight'] = contrast
            pseudo_trials.loc[i, 'stim_side'] = position
        pseudo_trials['signed_contrast'] = pseudo_trials['contrastRight']
        pseudo_trials.loc[pseudo_trials['signed_contrast'].isnull(),
                          'signed_contrast'] = -pseudo_trials['contrastLeft']
        pseudo_trials['choice'] = 1  # choice padding
        side, stim, act, _ = mut.format_data(pseudo_trials)
        msub_pseudo_tvec = optimal_Bayesian(act.values, stim, side.values)
        if take_out_unbiased:
            target_pLeft.append(msub_pseudo_tvec[(pseudo_trials.probabilityLeft != 0.5).values])
        else:
            target_pLeft.append(msub_pseudo_tvec)
    target_pLeft = np.concatenate(target_pLeft)
    target_pLeft = np.concatenate([target_pLeft, 1 - target_pLeft])
    out = np.histogram(target_pLeft, bins=np.arange(0, 1, bin_size_kde) + bin_size_kde/2., density=True)
    return out, target_pLeft


def check_bhv_fit_exists(subject, model, eids, resultpath):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    trainmeth = 'MCMC'  # This needs to be un-hard-coded if charles changes to diff. methods
    trunc_eids = [eid.split('-')[0] for eid in eids]
    str_sessionuuids = '_'.join(f'sess{k + 1}_{eid}' for k, eid in enumerate(trunc_eids))
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    modstr = modeldispatcher[model]
    filen = f'model_{modstr}_train_{trainmeth}_train_' + str_sessionuuids + '.pkl'
    fullpath = subjmodpath.joinpath(filen)
    return os.path.exists(fullpath), fullpath


def generate_imposter_session(imposterdf, eid, trialsdf, nbSampledSess=50, pLeftChange_when_stitch=True):
    """

    Parameters
    ----------
    imposterd: all sessions concatenated in pandas dataframe (generated with pipelines/03_generate_imposter_df.py)
    eid: eid of session of interest
    trialsdf: dataframe of trials of interest
    nbSampledSess: number of imposter sessions sampled to generate the final imposter session. NB: the length
    of the nbSampledSess stitched sessions must be greater than the session of interest, so typically choose a large
    number. If that condition is not verified a ValueError will be raised.
    Returns
    -------
    imposter session df

    """
    # this is to correct for when the eid is not part of the imposterdf eids
    # which is very possible when using imposter sessions from biaisChoice world.
    temp_trick = list(imposterdf[imposterdf.eid == eid].template_sess.unique())
    temp_trick.append(-1)
    template_sess_eid = temp_trick[0] 

    imposter_eids = np.random.choice(imposterdf[imposterdf.template_sess != template_sess_eid].eid.unique(),
                                     size=nbSampledSess,
                                     replace=False)
    sub_imposterdf = imposterdf[imposterdf.eid.isin(imposter_eids)].reset_index(drop=True)
    sub_imposterdf['row_id'] = sub_imposterdf.index
    sub_imposterdf['sorted_eids'] = sub_imposterdf.apply(lambda x: (np.argmax(imposter_eids == x['eid']) *
                                                                    sub_imposterdf.index.size + x.row_id),
                                                         axis=1)
    if np.any(sub_imposterdf['sorted_eids'].unique() != sub_imposterdf['sorted_eids']):
        raise ValueError('There is most probably a bug in the function')
    sub_imposterdf = sub_imposterdf.sort_values(by=['sorted_eids'])
    # seems to work better when starting the imposter session as the actual session, with an unbiased block
    sub_imposterdf = sub_imposterdf[(sub_imposterdf.probabilityLeft != 0.5) |
                     (sub_imposterdf.eid == imposter_eids[0])].reset_index(drop=True)
    if pLeftChange_when_stitch:
        valid_imposter_eids, current_last_pLeft = [], 0
        for i, imposter_eid in enumerate(imposter_eids):
            #  get first pLeft
            first_pLeft = sub_imposterdf[(sub_imposterdf.eid == imposter_eid)].probabilityLeft.values[0]
            #  make it such that stitches correspond to pLeft changepoints
            if np.abs(first_pLeft - current_last_pLeft) > 1e-8:
                valid_imposter_eids.append(imposter_eid)  # if first pLeft is different from current pLeft, accept sess
                #  take out the last block on the first session to stitch as it may not be a block with the right
                #  statistics (given the mouse stops the task there)
                second2last_pLeft = 1 - sub_imposterdf[(sub_imposterdf.eid == imposter_eid)].probabilityLeft.values[-1]
                second2last_block_idx = sub_imposterdf[(sub_imposterdf.eid == imposter_eid) &
                                                       (np.abs(sub_imposterdf.probabilityLeft -
                                                               second2last_pLeft) < 1e-8)].index[-1]
                last_block_idx = sub_imposterdf[(sub_imposterdf.eid == imposter_eid)].index[-1]
                sub_imposterdf = sub_imposterdf.drop(np.arange(second2last_block_idx + 1, last_block_idx + 1))
                #  update current last pLeft
                current_last_pLeft = sub_imposterdf[(sub_imposterdf.eid == imposter_eid)].probabilityLeft.values[-1]
                if np.abs(second2last_pLeft - current_last_pLeft) > 1e-8:
                    raise ValueError('There is most certainly a bug here')
        sub_imposterdf = sub_imposterdf[sub_imposterdf.eid.isin(valid_imposter_eids)].sort_values(by=['sorted_eids'])
        if sub_imposterdf.index.size < trialsdf.index.size:
            raise ValueError('you did not stitch enough imposter sessions. Simply increase the nbSampledSess argument')
        sub_imposterdf = sub_imposterdf.reset_index(drop=True)
    # select a random first block index <- this doesn't seem to work well, it changes the block statistics too much
    # idx_chge = np.where(sub_imposterdf.probabilityLeft.values[1:] != sub_imposterdf.probabilityLeft.values[:-1])[0]+1
    # random_number = np.random.choice(idx_chge[idx_chge < (sub_imposterdf.index.size - trialsdf.index.size)])
    # imposter_sess = sub_imposterdf.iloc[random_number:(random_number + trialsdf.index.size)].reset_index(drop=True)
    imposter_sess = sub_imposterdf.iloc[:trialsdf.index.size].reset_index(drop=True)
    return imposter_sess


def fit_load_bhvmod(target, subject, savepath, eids_train, eid_test, remove_old=False,
                    modeltype=expSmoothing_prevAction, one=None, behavior_data_train=None, beh_data_test=None):
    '''
    load/fit a behavioral model to compute target on a single session
    Params:
        eids_train: list of eids on which we train the network
        eid_test: eid on which we want to compute the target signals, only one string
        beh_data_test: if you have to launch the model on beh_data_test.
                       if beh_data_test is explicited, the eid_test will not be considered
        target can be pLeft or signcont. If target=pLeft, it will return the prior predicted by modeltype
                                         if modetype=None, then it will return the actual pLeft (.2, .5, .8)
    '''
    one = one or ONE()

    # check if is trained
    istrained, fullpath = check_bhv_fit_exists(subject, modeltype, eids_train, savepath)

    # load test session is beh_data_test is None
    if beh_data_test is None:
        beh_data_test = mut.load_session(eid_test, one=one)

    if target == 'signcont':
        if 'signedContrast' in beh_data_test.keys():
            out = beh_data_test['signedContrast']
        else:
            out = np.nan_to_num(beh_data_test['contrastLeft']) - np.nan_to_num(beh_data_test['contrastRight'])
        return out
    elif (target == 'pLeft') and (modeltype is None):
        return np.array(beh_data_test['probabilityLeft'])
    elif (target == 'pLeft') and (modeltype is optimal_Bayesian):  # bypass fitting and generate priors
        side, stim, act, _ = mut.format_data(beh_data_test)
        if isinstance(side, np.ndarray) and isinstance(act, np.ndarray):
            signal = optimal_Bayesian(act, stim, side)
        else:
            signal = optimal_Bayesian(act.values, stim, side.values)
        return signal.numpy().squeeze()

    if (not istrained) and (target != 'signcont') and (modeltype is not None):
        datadict = {'stim_side': [], 'actions': [], 'stimuli': []}
        for eid in eids_train:
            if behavior_data_train is None:
                data = mut.load_session(eid, one=one)
                if data['choice'] is None:
                    raise ValueError('Session choices produced are None. Debug models.utils.load_session,'
                                     f' or remove the eid {eid} from your input list.')
                stim_side, stimuli, actions, _ = mut.format_data(data)
            else:
                subdf = behavior_data_train[behavior_data_train.eid == eid]
                stim_side, stimuli, actions = subdf.stim_side.values, subdf.signedContrast.values, subdf.choice.values
            datadict['stim_side'].append(stim_side)
            datadict['stimuli'].append(stimuli)
            datadict['actions'].append(actions)
        stimuli, actions, stim_side = mut.format_input(datadict['stimuli'], datadict['actions'],
                                                       datadict['stim_side'])
        eids = np.array(eids_train)
        model = modeltype(savepath, eids, subject,
                          actions, stimuli, stim_side)
        model.load_or_train(remove_old=remove_old)
    elif (target != 'signcont') and (modeltype is not None):
        model = modeltype(savepath, eids_train, subject, actions=None, stimuli=None,
                          stim_side=None)
        model.load_or_train(loadpath=str(fullpath))

    # compute signal
    stim_side, stimuli, actions, _ = mut.format_data(beh_data_test)
    stimuli, actions, stim_side = mut.format_input([stimuli], [actions], [stim_side])
    signal = model.compute_signal(signal='prior' if target == 'pLeft' else target,
                                  act=actions,
                                  stim=stimuli,
                                  side=stim_side)['prior' if target == 'pLeft' else target]

    return signal.squeeze()


def remap_region(ids, source='Allen-lr', dest='Beryl-lr', output='acronym', br=None):
    br = br or BrainRegions()
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']
    elif output == 'name':
        return br.get(br.id[br.mappings[dest][inds]])['name']
    else:
        return br.get(br.id[br.mappings[dest][inds]])


def compute_target(target, subject, eids_train, eid_test, savepath,
                   modeltype=expSmoothing_prevAction, one=None, behavior_data_train=None,
                   beh_data_test=None):
    """
    Computes regression target for use with regress_target, using subject, eid, and a string
    identifying the target parameter to output a vector of N_trials length containing the target

    Parameters
    ----------
    target : str
        String in ['prior', 'prederr', 'signcont'], indication model-based prior, prediction error,
        or simple signed contrast per trial
    subject : str
        Subject identity in the IBL database, e.g. KS022
    eids_train : list of str
        list of UUID identifying sessions on which the model is trained.
    eids_test : str
        UUID identifying sessions on which the target signal is computed
    savepath : str
        where the beh model outputs are saved
    behmodel : str
        behmodel to use
    pseudo : bool
        Whether or not to compute a pseudosession result, rather than a real result.
    modeltype : behavior_models model object
        Instantiated object of behavior models. Needs to be instantiated for pseudosession target
        generation in the case of a 'prior' or 'prederr' target.
    beh_data : behavioral data feed to the model when using pseudo-sessions

    Returns
    -------
    pandas.Series
        Pandas series in which index is trial number, and value is the target
    """
    if target not in possible_targets:
        raise ValueError('target should be in {}'.format(possible_targets))

    tvec = fit_load_bhvmod(target, subject, savepath.as_posix() + '/', eids_train, eid_test, remove_old=False,
                           modeltype=modeltype, one=one, behavior_data_train=behavior_data_train,
                           beh_data_test=beh_data_test)

    # todo make pd.Series
    return tvec


def regress_target(tvec, binned, estimatorObject, estimator_kwargs, use_openturns, target_distribution, bin_size_kde,
                   balanced_continuous_target=True, hyperparam_grid=None, test_prop=0.2, nFolds=5, save_binned=False,
                   verbose=False, shuffle=True, outer_cv=True, balanced_weight=False,
                   normalize_input=False, normalize_output=False):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator

    Parameters
    ----------
    normalize_output: gives the possibility of normalizing (take out the mean across trials) of the output
    (what we want to predict)
    normalize_input: gives the possibility of normalizing (take out the mean across trials) of the input
    (the binned neural activity). Average is taken across trials for each unit (one average per unit is computed).
    tvec : pandas.Series
        Series in which trial number is the index and the value is the regression target
    binned : numpy.ndarray
        N_trials X N_neurons array, in which a single element is a spike count on a trial for a
        given neuron
    estimator : sklearn.linear_model estimator
        Estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
    hyperparam_grid : dict
        Dictionary with key indicating hyperparameter to grid search over, and value being a the
        nodes on the grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
        Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
    test_prop : float
        Proportion of data to hold out as the test set after running hyperparameter tuning.
        Default 0.2
    nFolds : int
        Number of folds for cross-validation during hyperparameter tuning.
    save_binned : bool
        Whether or not to put the regressors in binned into the output dictionary. Can cause file
        bloat if saving outputs.
    verbose : bool
        Whether you want to hear about the function's life, how things are going,
        and what the neighbor down the street said to it the other day.
    outer_cv: bool
        Perform outer cross validation such that the testing spans the entire dataset
    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Decoding intercept
            - Per-trial target values (copy of tvec)
            - Per-trial predictions from model
            - Input regressors (optional, see binned argument)
    """
    # initialize outputs
    Rsquareds_test, Rsquareds_train, weights, intercepts = [], [], [], []
    predictions, predictions_test, idxes_test, idxes_train, best_params = [], [], [], [], []

    # train / test split
    # Split the dataset in two equal parts
    # when shuffle=False, the method will take the end of the dataset to create the test set
    indices = np.arange(len(tvec))
    if outer_cv:
        outer_kfold = KFold(n_splits=nFolds, shuffle=shuffle).split(indices)
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if isinstance(estimatorObject, LinearModelCV):
        if hyperparam_grid is not None:
            raise TypeError('If using a CV estimator hyperparam_grid will not be respected;'
                            ' set to None')
        cvest = True
        estimatorObject.cv = nFolds  # Overwrite user spec to make sure nFolds is used
        clf = estimatorObject
        raise NotImplemented('the code does not support a CV-type estimator for the moment.')
    else:
        cvest = False
        for train_index, test_index in outer_kfold:
            X_train, X_test = binned[train_index], binned[test_index]
            y_train, y_test = tvec[train_index], tvec[test_index]

            idx_inner = np.arange(len(X_train))
            inner_kfold = KFold(n_splits=nFolds, shuffle=shuffle).split(idx_inner)

            r2s = np.zeros([nFolds, len(hyperparam_grid['alpha'])])
            for ifold, (train_inner, test_inner) in enumerate(inner_kfold):
                X_train_inner, X_test_inner = X_train[train_inner], X_train[test_inner]
                y_train_inner, y_test_inner = y_train[train_inner], y_train[test_inner]

                # normalization when necessary
                mean_X_train_inner = X_train_inner.mean(axis=0) if normalize_input else 0
                X_train_inner = X_train_inner - mean_X_train_inner
                X_test_inner = X_test_inner - mean_X_train_inner
                mean_y_train_inner = y_train_inner.mean(axis=0) if normalize_output else 0
                y_train_inner = y_train_inner - mean_y_train_inner

                for i_alpha, alpha in enumerate(hyperparam_grid['alpha']):
                    estimator = estimatorObject(**{**estimator_kwargs, 'alpha': alpha})
                    if balanced_weight:
                        estimator.fit(X_train_inner, y_train_inner,
                                      sample_weight=balanced_weighting(vec=y_train_inner,
                                                                       continuous=balanced_continuous_target,
                                                                       use_openturns=use_openturns,
                                                                       bin_size_kde=bin_size_kde,
                                                                       target_distribution=target_distribution))
                    else:
                        estimator.fit(X_train_inner, y_train_inner)
                    pred_test_inner = estimator.predict(X_test_inner) + mean_y_train_inner
                    r2s[ifold, i_alpha] = r2_score(y_test_inner, pred_test_inner)

            r2s_avg = r2s.mean(axis=0)
            best_alpha = hyperparam_grid['alpha'][np.argmax(r2s_avg)]
            clf = estimatorObject(**{**estimator_kwargs, 'alpha': best_alpha})

            # normalization when necessary
            mean_X_train = X_train.mean(axis=0) if normalize_input else 0
            X_train = X_train - mean_X_train
            mean_y_train = y_train.mean(axis=0) if normalize_output else 0
            y_train = y_train - mean_y_train

            if balanced_weight:
                clf.fit(X_train, y_train, sample_weight=balanced_weighting(vec=y_train,
                                                                           continuous=continuous_target,
                                                                           use_openturns=use_openturns,
                                                                           bin_size_kde=bin_size_kde,
                                                                           target_distribution=target_distribution))
            else:
                clf.fit(X_train, y_train)

            # compute R2 on the train data
            y_pred_train = clf.predict(X_train)
            Rsquareds_train.append(r2_score(y_train + mean_y_train, y_pred_train + mean_y_train))

            # compute R2 on held-out data
            y_true, prediction = y_test, clf.predict(binned - mean_X_train) + mean_y_train
            Rsquareds_test.append(r2_score(y_true, prediction[test_index]))

            # prediction, target, idxes_test, idxes_train
            predictions.append(prediction)
            predictions_test.append(prediction[test_index])
            idxes_test.append(test_index)
            idxes_train.append(train_index)
            weights.append(clf.coef_)
            if clf.fit_intercept:
                intercepts.append(clf.intercept_)
            else:
                intercepts.append(None)
            best_params.append({'alpha': best_alpha})

    full_test_prediction = np.zeros(len(tvec))
    for k in range(nFolds):
        full_test_prediction[idxes_test[k]] = predictions_test[k]

    outdict = dict()
    outdict['Rsquared_test_full'] = r2_score(tvec, full_test_prediction)
    outdict['Rsquareds_train'] = Rsquareds_train
    outdict['Rsquareds_test'] = Rsquareds_test
    outdict['weights'] = weights
    outdict['intercepts'] = intercepts
    outdict['target'] = tvec
    outdict['predictions'] = predictions
    outdict['predictions_test'] = predictions_test
    outdict['idxes_test'] = idxes_test
    outdict['idxes_train'] = idxes_train
    outdict['best_params'] = best_params
    outdict['nFolds'] = nFolds
    if save_binned:
        outdict['regressors'] = binned

    # logging
    if verbose:
        # verbose output
        if outer_cv:
            print('Performance is only describe for last outer fold \n')
        print("Possible regularization parameters over {} validation sets:".format(nFolds))
        print('{}: {}'.format(list(hyperparam_grid.keys())[0], hyperparam_grid))
        print("\nBest parameters found over {} validation sets:".format(nFolds))
        print(clf.best_params_)
        print("\nAverage scores over {} validation sets:".format(nFolds))
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n", "Detailed scores on {} validation sets:".format(nFolds))
        for i_fold in range(nFolds):
            tscore_fold = list(np.round(clf.cv_results_['split{}_test_score'.format(int(i_fold))],
                                        3))
            print("perf on fold {}: {}".format(int(i_fold), tscore_fold))

        print("\n", "Detailed classification report:", "\n")
        print("The model is trained on the full (train + validation) set.")
        # print("\n", "Rsquare on held-out test data: {}".format(np.round(Rsquared_test, 3)), "\n")

        '''
        import pickle
        outdict_verbose = dict()
        outdict_verbose['binned_activity'] = binned
        outdict_verbose['labels'] = tvec
        outdict_verbose['pred_train'] = y_pred_train
        outdict_verbose['R2_train'] = Rsquared_train
        outdict_verbose['pred_test'] = y_pred
        outdict_verbose['R2_test'] = Rsquared_test
        outdict_verbose['regul_term'] = clf.best_params_
        pickle.dump(outdict_verbose, open('eid_{}_sanity.pkl'.format(eid), 'wb'))
        '''

    return outdict


def pdf_from_histogram(x, out):
    return out[0][(x[:, None] > out[1][None]).sum(axis=-1) - 1]


def balanced_weighting(vec, continuous, use_openturns, bin_size_kde, target_distribution):
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.KernelSmoothing.html?highlight=kernel%20smoothing
    # This plug-in method for bandwidth estimation is based on the solve-the-equation rule from (Sheather, Jones, 1991).
    if continuous:
        if use_openturns:
            factory = openturns.KernelSmoothing()
            sample = openturns.Sample(vec[:, None])
            bandwidth = factory.computePluginBandwidth(sample)
            distribution = factory.build(sample, bandwidth)
            proposal_weights = np.array(distribution.computePDF(sample)).squeeze()
            balanced_weight = np.ones(vec.size) / proposal_weights
        else:
            emp_distribution = np.histogram(vec, bins=np.arange(0, 1, bin_size_kde) + bin_size_kde/2, density=True)
            balanced_weight = pdf_from_histogram(vec, target_distribution)/pdf_from_histogram(vec, emp_distribution)
        #  plt.hist(y_train_inner[:, None], density=True)
        #  plt.plot(sample, proposal_weights, '+')
    else:
        balanced_weight = compute_sample_weight("balanced", y=vec)
    return balanced_weight


def return_regions(eid, sessdf, QC_CRITERIA=1, NUM_UNITS=10):
    df_insertions = sessdf.loc[sessdf['eid'] == eid]
    brainreg = BrainRegions()
    my_regions = {}
    for i, ins in tqdm(df_insertions.iterrows(), desc='Probe: ', leave=False):
        probe = ins['probe']
        spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
        clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
        beryl_reg = remap_region(clusters.atlas_id, br=brainreg)
        qc_pass = (clusters['label'] >= QC_CRITERIA).values
        regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        probe_regions = []
        for region in tqdm(regions, desc='Region: ', leave=False):
            reg_mask = (beryl_reg == region)
            reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            if len(reg_clu_ids) > NUM_UNITS:
                probe_regions.append(region)
        my_regions[probe] = probe_regions
    return my_regions


# %% Define helper functions for dask workers to use
def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, N,
                        output_path, time_window, today, target, add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([today, region, 'target', target,
                   'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_'),
                   'pseudo_id', str(pseudo_id), add_to_saving_path]) + '.pkl'
    for folder in [subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    outdict = {'fit': fit_result, 'pseudo_id': pseudo_id,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region, 'N_units': N}
    fw = open(probefolder.joinpath(fn), 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


def optimal_Bayesian(act, stim, side):
    '''
    Generates the optimal prior
    Params:
        act (array of shape [nb_sessions, nb_trials]): action performed by the mice of shape
        side (array of shape [nb_sessions, nb_trials]): stimulus side (-1 (right), 1 (left)) observed by the mice
    Output:
        prior (array of shape [nb_sessions, nb_chains, nb_trials]): prior for each chain and session
    '''
    act = torch.from_numpy(act)
    side = torch.from_numpy(side)
    lb, tau, ub, gamma = 20, 60, 100, 0.8
    nb_blocklengths = 100
    nb_typeblocks = 3
    eps = torch.tensor(1e-15)

    alpha = torch.zeros([act.shape[-1], nb_blocklengths, nb_typeblocks])
    alpha[0, 0, 1] = 1
    alpha = alpha.reshape(-1, nb_typeblocks * nb_blocklengths)
    h = torch.zeros([nb_typeblocks * nb_blocklengths])

    # build transition matrix
    b = torch.zeros([nb_blocklengths, nb_typeblocks, nb_typeblocks])
    b[1:][:, 0, 0], b[1:][:, 1, 1], b[1:][:, 2, 2] = 1, 1, 1  # case when l_t > 0
    b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = 1, 1, 1. / 2  # case when l_t = 1
    n = torch.arange(1, nb_blocklengths + 1)
    ref = torch.exp(-n / tau) * (lb <= n) * (ub >= n)
    torch.flip(ref.double(), (0,))
    hazard = torch.cummax(ref / torch.flip(torch.cumsum(torch.flip(ref.double(), (0,)), 0) + eps, (0,)), 0)[0]
    l = torch.cat((torch.unsqueeze(hazard, -1),
                   torch.cat((torch.diag(1 - hazard[:-1]),
                              torch.zeros(nb_blocklengths - 1)[None]),
                             axis=0)), axis=-1)  # l_{t-1}, l_t
    transition = eps + torch.transpose(l[:, :, None, None] * b[None], 1, 2).reshape(nb_typeblocks * nb_blocklengths, -1)

    # likelihood
    lks = torch.hstack([gamma * (side[:, None] == -1) + (1 - gamma) * (side[:, None] == 1),
                        torch.ones_like(act[:, None]) * 1. / 2,
                        gamma * (side[:, None] == 1) + (1 - gamma) * (side[:, None] == -1)])
    to_update = torch.unsqueeze(torch.unsqueeze(act.not_equal(0), -1), -1) * 1

    for i_trial in range(act.shape[-1]):
        # save priors
        if i_trial > 0:
            alpha[i_trial] = torch.sum(torch.unsqueeze(h, -1) * transition, axis=0) * to_update[i_trial - 1] \
                             + alpha[i_trial - 1] * (1 - to_update[i_trial - 1])
        h = alpha[i_trial] * lks[i_trial].repeat(nb_blocklengths)
        h = h / torch.unsqueeze(torch.sum(h, axis=-1), -1)

    predictive = torch.sum(alpha.reshape(-1, nb_blocklengths, nb_typeblocks), 1)
    Pis = predictive[:, 0] * gamma + predictive[:, 1] * 0.5 + predictive[:, 2] * (1 - gamma)

    return 1 - Pis


def return_path(eid, sessdf, pseudo_ids=[-1], **kwargs):
    """
    Parameters
    ----------
    single_region: Bool, decoding using region wise or pulled over regions
    eid: eid of session
    sessdf: dataframe of session eid
    pseudo_id: whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered.
    can not be 0
    nb_runs: nb of independent runs performed. this was added after consequent variability was observed across runs.
    modelfit_path: outputs of behavioral fits
    output_path: outputs of decoding fits
    one: ONE object -- this is not to be used with dask, this option is given for debugging purposes
    """

    df_insertions = sessdf.loc[sessdf['eid'] == eid]
    subject = df_insertions['subject'].to_numpy()[0]
    brainreg = BrainRegions()

    filenames = []
    if kwargs['merged_probes']:
        across_probes = {'regions': [], 'clusters': [], 'times': [], 'qc_pass': []}
        for _, ins in df_insertions.iterrows():
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = remap_region(clusters.atlas_id, br=brainreg)
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            across_probes['regions'].extend(beryl_reg)
            across_probes['qc_pass'].extend(qc_pass)
        across_probes = {k: np.array(v) for k, v in across_probes.items()}
        # warnings.filterwarnings('ignore')
        if kwargs['single_region']:
            regions = [[k] for k in np.unique(across_probes['regions'])]
        else:
            regions = [np.unique(across_probes['regions'])]
        df_insertions_iterrows = pd.DataFrame.from_dict({'1': 'mergedProbes'},
                                                        orient='index',
                                                        columns=['probe']).iterrows()
    else:
        df_insertions_iterrows = df_insertions.iterrows()

    for i, ins in df_insertions_iterrows:
        probe = ins['probe']
        if not kwargs['merged_probes']:
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = remap_region(clusters.atlas_id, br=brainreg)
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            regions = np.unique(beryl_reg)
        for region in regions:
            if kwargs['merged_probes']:
                reg_mask = np.isin(across_probes['regions'], region)
                reg_clu_ids = np.argwhere(reg_mask & across_probes['qc_pass']).flatten()
            else:
                reg_mask = beryl_reg == region
                reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            N_units = len(reg_clu_ids)
            if N_units < kwargs['min_units']:
                continue

            for pseudo_id in pseudo_ids:
                filenames.append(save_region_results(None, pseudo_id, subject, eid, probe,
                                                     str(np.squeeze(region)) if kwargs[
                                                         'single_region'] else 'allRegions',
                                                     N_units, output_path=kwargs['output_path'],
                                                     time_window=kwargs['time_window'],
                                                     today=kwargs['today'],
                                                     compute=False))
    return filenames


possible_targets = ['prederr', 'signcont', 'pLeft']

modeldispatcher = {expSmoothing_prevAction: 'expSmoothingPrevActions',
                   expSmoothing_stimside: 'expSmoothingStimSides',
                   biased_ApproxBayesian: 'biased_Approxbayesian',
                   biased_Bayesian: 'biased_Bayesian',
                   optimal_Bayesian: 'optimal_bayesian',
                   None: 'oracle'
                   }
