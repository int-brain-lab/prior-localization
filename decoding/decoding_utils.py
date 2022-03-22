import os
import numpy as np
import pandas as pd
from models import utils as mut
from pathlib import Path
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from models.expSmoothing_stimside import expSmoothing_stimside
from models.biasedApproxBayesian import biased_ApproxBayesian
from models.biasedBayesian import biased_Bayesian
from models.optimalBayesian import optimal_Bayesian
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model._coordinate_descent import LinearModelCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

possible_targets = ['prior', 'prederr', 'signcont', 'pLeft',
                    'choice','feedback']

modeldispatcher = {expSmoothing_prevAction: 'expSmoothingPrevActions',
                   expSmoothing_stimside: 'expSmoothingStimSides',
                   biased_ApproxBayesian: 'biased_Approxbayesian',
                   biased_Bayesian: 'biased_Bayesian',
                   optimal_Bayesian: 'optimal_bayesian',
                   None: 'none'
                   }

# Loading data and input utilities


def decoding_details(TARGET,MODEL,SCORE,
                     ESTIMATORSTR,
                     ALIGN_TIME,
                     CONTROL_FEATURES,
                     N_PSEUDO,NULL_TYPE,TIME_WINDOW,
                     ADD_TO_SAVING_PATH,
                     USE_FAKE_DATA=False):
    '''
    MODEL must be in modeldispatcher in decoding_utils
    '''
    
    start_tw, end_tw = TIME_WINDOW
    
    details = '_'.join(['decode', TARGET,
                   modeldispatcher[MODEL] if TARGET in ['prior', 'prederr'] else 'task',
                   ESTIMATORSTR, SCORE,
                   'control', *CONTROL_FEATURES,
                   str(N_PSEUDO), NULL_TYPE,
                   'align', ALIGN_TIME, 
                   'timeWin', 
                   str(start_tw).replace('.', '_'), 
                   str(end_tw).replace('.', '_')])
    if USE_FAKE_DATA:
        details = details + '_fake'
    if not (ADD_TO_SAVING_PATH == ''):
        details = details + '_' + ADD_TO_SAVING_PATH
    return details


def query_sessions(selection='all'):
    '''
    Filters sessions on some canonical filters
    returns dataframe with index being EID, so indexing results in subject name and probe
    identities in that EID.
    '''
    one = ONE()
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

    # Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    retdf = pd.DataFrame({'subject': all_subjects, 'eid': all_eids, 'probe': all_probes})
    retdf.sort_values('subject', inplace=True)
    return retdf


def check_bhv_fit_exists(subject, model, eids, resultpath):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    trainmeth = 'MCMC'  # This needs to be un-hard-coded if charles changes to diff. methods
    trunc_eids = [eid.split('-')[0] for eid in eids]
    str_sessionuuids = '_'.join(f'sess{k+1}_{eid}' for k, eid in enumerate(trunc_eids))
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    modstr = modeldispatcher[model]
    filen = f'model_{modstr}_train_{trainmeth}_train_' + str_sessionuuids + '.pkl'
    fullpath = subjmodpath.joinpath(filen)
    return os.path.exists(fullpath), fullpath


def fit_load_bhvmod(target, subject, savepath, eids_train, eid_test, remove_old=False,
                    modeltype=expSmoothing_prevAction, one=None,
                    beh_data_test=None):
    '''
    load/fit a behavioral model to compute target on a single session
    Params:
        eids_train: list of eids on which we train the network
        eid_test: eid on which we want to compute the target signals, only one string
        beh_data_test: if you have to launch the model on beh_data_test
    '''

    one = one or ONE()

    # check if is trained
    istrained, fullpath = check_bhv_fit_exists(subject, modeltype, eids_train, savepath)

    if (beh_data_test is not None) and (not istrained) and (target not in ['signcont', 'pLeft','choice','feedback']):
        raise ValueError('when actions, stimuli and stim_side are all defined,'
                         ' the model must have been trained')

    if (not istrained) and (target not in ['signcont', 'pLeft','choice','feedback']):
        datadict = {'stim_side': [], 'actions': [], 'stimuli': []}
        for eid in eids_train:
            data = mut.load_session(eid, one=one)
            if data['choice'] is None:
                raise ValueError('Session choices produced are None.'
                                 'Debug models.utils.load_session,'
                                 f' or remove the eid {eid} from your input list.')
            stim_side, stimuli, actions, _ = mut.format_data(data)
            datadict['stim_side'].append(stim_side)
            datadict['stimuli'].append(stimuli)
            datadict['actions'].append(actions)
        stimuli, actions, stim_side = mut.format_input(datadict['stimuli'], datadict['actions'],
                                                       datadict['stim_side'])
        eids = np.array(eids_train)
        model = modeltype(savepath, eids, subject,
                          actions, stimuli, stim_side)
        model.load_or_train(remove_old=remove_old)
    elif target not in ['signcont', 'pLeft','choice','feedback']:
        model = modeltype(savepath, eids_train, subject, actions=None, stimuli=None,
                          stim_side=None)
        model.load_or_train(loadpath=str(fullpath))

    # load test session
    if beh_data_test is None:
        beh_data_test = mut.load_session(eid_test, one=one)

    if target == 'signcont':
        out = np.nan_to_num(beh_data_test['contrastLeft']) - \
            np.nan_to_num(beh_data_test['contrastRight'])
        return out
    elif target == 'pLeft':
        return np.array(beh_data_test['probabilityLeft'])
    elif target == 'choice':
        return np.array(beh_data_test['choice'])
    elif target == 'feedback':
        return np.array(beh_data_test['feedbackType'])

    # compute signal
    stim_side, stimuli, actions, _ = mut.format_data(beh_data_test)
    stimuli, actions, stim_side = mut.format_input([stimuli], [actions], [stim_side])
    signal = model.compute_signal(signal=target, act=actions, stim=stimuli, side=stim_side)[target]

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
                   modeltype=expSmoothing_prevAction, one=None,
                   beh_data=None):
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

    target = fit_load_bhvmod(target, subject, savepath, eids_train, eid_test, remove_old=False,
                             modeltype=modeltype, one=one, beh_data_test=beh_data)

    # todo make pd.Series
    return target


def regress_target(tvec, binned, estimatorObject, estimator_kwargs,
                   hyperparam_grid=None, test_prop=0.2, nFolds=5, save_binned=False,
                   verbose=False, shuffle=True, outer_cv=True, 
                   balanced_weight=False, 
                   control_features=[],
                   SCORE='r2'):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator

    Parameters
    ----------
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
    SCORE: str
        metric used to quantify regression performance.
        used to choose the best hyper parameters during cross validation. 
        if 'accuracy', then output dictionary contains regression probabilities.
        assumed that there are only two classes. more than two is not implemented
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
    Scores_test, Scores_train, weights, intercepts = [], [], [], []
    predictions, predictions_test, idxes_test, idxes_train, best_params = [], [], [], [], []
    if SCORE == 'accuracy':
        probabilities, probabilities_test = [], []
        current_score = lambda *args: accuracy_score(*args)
    elif SCORE == 'r2':
        current_score = lambda *args: r2_score(*args)
    else:
        raise TypeError('SCORE must be accuracy or r2, \
                        others are not implemented')

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
        classes = []
        for train_index, test_index in outer_kfold:
            X_train, X_test = binned[train_index], binned[test_index]
            y_train, y_test = tvec[train_index], tvec[test_index]

            idx_inner = np.arange(len(X_train))
            inner_kfold = KFold(n_splits=nFolds, shuffle=shuffle).split(idx_inner)
            
            try:
                hyperkey = list(hyperparam_grid.keys())
                assert len(hyperkey)==1
                hyperkey = hyperkey[0]
            except AssertionError:
                raise AssertionError('too many hyper parameters, only 1 allowed')
            scores = np.zeros([nFolds, len(hyperparam_grid[hyperkey])])
            for ifold, (train_inner, test_inner) in enumerate(inner_kfold):
                X_train_inner, X_test_inner = X_train[train_inner], X_train[test_inner]
                y_train_inner, y_test_inner = y_train[train_inner], y_train[test_inner]

                for i_alpha, alpha in enumerate(hyperparam_grid[hyperkey]):
                    estimator = estimatorObject(**{**estimator_kwargs, hyperkey: alpha})
                    if balanced_weight:
                        estimator.fit(X_train_inner, y_train_inner, sample_weight=compute_sample_weight("balanced",
                                                                                                    y=y_train_inner))
                    else:
                        estimator.fit(X_train_inner, y_train_inner)
                    pred_test_inner = estimator.predict(X_test_inner)
                    scores[ifold, i_alpha] = current_score(y_test_inner, pred_test_inner)
                        

            scores_avg = scores.mean(axis=0)
            best_alpha = hyperparam_grid[hyperkey][np.argmax(scores_avg)]
            clf = estimatorObject(**{**estimator_kwargs, hyperkey: best_alpha})
            if balanced_weight:
                clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y=y_train))
            else:
                clf.fit(X_train, y_train)

            # compute score on the train data
            y_pred_train = clf.predict(X_train)
            Scores_train.append(current_score(y_train, y_pred_train))

            # compute score on held-out data
            y_true, y_pred = y_test, clf.predict(X_test)
            Scores_test.append(current_score(y_true, y_pred))

            # prediction, target, idxes_test, idxes_train
            predictions.append(clf.predict(binned))
            predictions_test.append(clf.predict(binned)[test_index])
            if SCORE == 'accuracy':
                probabilities.append(clf.predict_proba(binned))
                probabilities_test.append(clf.predict_proba(binned)[test_index])
                classes.append(clf.classes_)
            idxes_test.append(test_index)
            idxes_train.append(train_index)
            weights.append(clf.coef_)
            if clf.fit_intercept:
                intercepts.append(clf.intercept_)
            else:
                intercepts.append(None)
            best_params.append({hyperkey:best_alpha})
            
    full_test_prediction = np.zeros(len(tvec))
    for k in range(nFolds):
        full_test_prediction[idxes_test[k]] = predictions_test[k]

    outdict = dict()
    outdict['Score_test_full'] = current_score(tvec, full_test_prediction)
    outdict['Scores_train'] = Scores_train
    outdict['Scores_test'] = Scores_test
    outdict['weights'] = weights
    outdict['intercepts'] = intercepts
    outdict['target'] = tvec
    outdict['predictions'] = predictions
    outdict['predictions_test'] = predictions_test
    if SCORE == 'accuracy':
        outdict['probabilities'] = probabilities
        outdict['probabilities_test'] = probabilities_test
        outdict['classes'] = classes
    outdict['idxes_test'] = idxes_test
    outdict['idxes_train'] = idxes_train
    outdict['best_params'] = best_params
    outdict['nFolds'] = nFolds
    if save_binned:
        outdict['regressors'] = binned
    # if isin:
    #     outdict['predictions_continuous'] = predictions_continuous
    #     outdict['predictions_continuous_test'] = predictions_continuous_test

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
        print("\n", "Score on held-out test data: {}".format(np.round(Scores_test, 3)), "\n")

        '''
        import pickle
        outdict_verbose = dict()
        outdict_verbose['binned_activity'] = binned
        outdict_verbose['labels'] = tvec
        outdict_verbose['pred_train'] = y_pred_train
        outdict_verbose['Score_train'] = Scores_train
        outdict_verbose['pred_test'] = y_pred
        outdict_verbose['Score_test'] = Scores_test
        outdict_verbose['regul_term'] = clf.best_params_
        pickle.dump(outdict_verbose, open('eid_{}_sanity.pkl'.format(eid), 'wb'))
        '''

    return outdict

def get_impostor_target(targets, labels, current_label=None,
                        seed_idx=None, verbose=False):
    """
    Generate impostor targets by selecting from a list of current targets of variable length.
    Targets are selected and stitched together to the length of the current labeled target,
    aka 'Frankenstein' targets, often used for evaluating a null distribution while decoding.
    Parameters
    ----------
    targets : list of all targets
            targets may be arrays of any dimension (a,b,...,z)
            but must have the same shape except for the last dimension, z.  All targets must
            have z > 0.
    labels : numpy array of strings
            labels corresponding to each target e.g. session eid.
            only targets with unique labels are used to create impostor target.  Typically,
            use eid as the label because each eid has a unique target.
    current_label : string
            targets with the current label are not used to create impostor
            target.  Size of corresponding target is used to determine size of impostor
            target.  If None, a random selection from the set of unique labels is used.
    Returns
    --------
    impostor_final : numpy array, same shape as all targets except last dimension
    """

    np.random.seed(seed_idx)

    unique_labels, unique_label_idxs = np.unique(labels, return_index=True)
    unique_targets = [targets[unique_label_idxs[i]] for i in range(len(unique_label_idxs))]
    if current_label is None:
        current_label = np.random.choice(unique_labels)
    avoid_same_label = ~(unique_labels == current_label)
    # current label must correspond to exactly one unique label
    assert len(np.nonzero(~avoid_same_label)[0]) == 1
    avoided_index = np.nonzero(~avoid_same_label)[0][0]
    nonavoided_indices = np.nonzero(avoid_same_label)[0]
    ntargets = len(nonavoided_indices)
    all_impostor_targets = [unique_targets[nonavoided_indices[i]] for i in range(ntargets)]
    all_impostor_sizes = np.array([all_impostor_targets[i].shape[-1] for i in range(ntargets)])
    current_target_size = unique_targets[avoided_index].shape[-1]
    if verbose:
        print('impostor target has length %s' % (current_target_size))
    assert np.min(all_impostor_sizes) > 0  # all targets must be nonzero in size
    max_needed_to_tile = int(np.max(all_impostor_sizes) / np.min(all_impostor_sizes)) + 1
    tile_indices = np.random.choice(np.arange(len(all_impostor_targets), dtype=int),
                                    size=max_needed_to_tile,
                                    replace=False)
    impostor_tiles = [all_impostor_targets[tile_indices[i]] for i in range(len(tile_indices))]
    impostor_tile_sizes = all_impostor_sizes[tile_indices]
    if verbose:
        print('Randomly chose %s targets to tile the impostor target' % (max_needed_to_tile))
        print('with the following sizes:', impostor_tile_sizes)

    number_of_tiles_needed = np.sum(np.cumsum(impostor_tile_sizes) < current_target_size) + 1
    impostor_tiles = impostor_tiles[:number_of_tiles_needed]
    if verbose:
        print('%s of %s needed to tile the entire impostor target' % (number_of_tiles_needed,
                                                                      max_needed_to_tile))

    impostor_stitch = np.concatenate(impostor_tiles, axis=-1)
    start_ind = np.random.randint((impostor_stitch.shape[-1] - current_target_size) + 1)
    impostor_final = impostor_stitch[..., start_ind:start_ind + current_target_size]
    if verbose:
        print('%s targets stitched together with shift of %s\n' % (number_of_tiles_needed,
                                                                   start_ind))

    np.random.seed(None)  # reset numpy seed to None

    return impostor_final