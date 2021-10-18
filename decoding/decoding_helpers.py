
from os.path import join, isfile
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from ibllib.atlas import BrainRegions
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from brainbox.numerical import ismember

def regress(population_activity, trial_targets, reg=None, 
            cross_validation=None, return_training_r2_weights=False):
    """
    Brandon got from Guido in 2021 and built on it
    Brandon, returns training weights of decoder
    
    Perform linear regression to predict a continuous variable from neural data
    Parameters
    ----------
    population_activity : 2D array (trials x neurons)
        population activity of all neurons in the population for each trial.
    trial_targets : 1D or 2D array
        the decoding target per trial as a continuous variable
    reg : scikit-learn regression object
        from sklearn.linear_model import LinearRegression
                regularization = LinearRegression()
                can include regularization as follows
                        None = no regularization using ordinary least squares linear regression
                        'L1' = L1 regularization using Lasso
                        'CVL1' = L1 regularization using LassoCV cross vaidated for regularization weight "alpha"
                        'L2' = L2 regularization using Ridge regression
    cross_validation : None or scikit-learn object
        which cross-validation method to use, for example 5-fold:
                    from sklearn.model_selection import KFold
                    cross_validation = KFold(n_splits=5)
    return_training_r2_weights : bool
        if set to True the classifier will also return the performance on the training set,
        the r2 score of predictions, the coefficients of regression and the intercept of regression
    Returns
    -------
    pred : 1D array
        array with predictions
    pred_training : 1D array
        array with predictions for the training set (only if return_training_r2_weights is True)
        if cross_validation is None, then pred_training = pred
    r2 : float
        array with predictions for the training set (only if return_training_r2_weights is True)
    coefs : 
        array with predictions for the training set (only if return_training_r2_weights is True)
    intercepts : 
        array with predictions for the training set (only if return_training_r2_weights is True)
    """
        
    # Check input
#     if (cross_validation is None) and (return_training_r2_weights is True):
#         raise RuntimeError('cannot return training accuracy without cross-validation')
    if population_activity.shape[0] != trial_targets.shape[0]:
        raise ValueError('trial_targets is not the same length as the first dimension of '
                         'population_activity')

    # Initialize weights saved
    coefs = []
    intercepts = []
    if return_training_r2_weights:
        pred_training = np.empty(trial_targets.shape[0])
    
    if cross_validation is None:
        # Fit the model on all the data
        reg.fit(population_activity, trial_targets)
        pred = reg.predict(population_activity)
    
        coefs.append(reg.coef_)
        intercepts.append(reg.intercept_)
        if return_training_r2_weights:
            pred_training = np.copy(pred)
    else:
        pred = np.empty(trial_targets.shape[0])
        
        for train_index, test_index in cross_validation.split(population_activity):
            # Fit the model to the training data
            reg.fit(population_activity[train_index], trial_targets[train_index])

            # Predict the held-out test data
            pred[test_index] = reg.predict(population_activity[test_index])

            # Predict the training data
            if return_training_r2_weights:
                pred_training[train_index] = reg.predict(population_activity[train_index])
            
            coefs.append(reg.coef_)
            intercepts.append(reg.intercept_)
    
    coefs, intercepts = np.column_stack(tuple(coefs)), np.column_stack(tuple(intercepts))
    if return_training_r2_weights:
        r2 = r2_score(trial_targets, pred)
        return pred, pred_training, r2, coefs, intercepts
    
    return pred, coefs, intercepts

def remap(ids, source='Allen', dest='Beryl', br=BrainRegions()):
    '''
    Guido's work
    '''
    _, inds = ismember(ids, br.id[br.mappings[source]])
    return br.id[br.mappings[dest][inds]]


def get_incl_trials(trials, target, excl_5050, min_rt):
    '''
    Guido's work
    '''
    incl_trials = np.ones(len(trials['choice'])).astype(bool)
    if excl_5050:
        incl_trials[trials['probabilityLeft'] == 0.5] = False
    if 'pos' in target:
        incl_trials[trials['feedbackType'] == -1] = False  # Exclude all rew. ommissions
    if 'neg' in target:
        incl_trials[trials['feedbackType'] == 1] = False  # Exclude all rewards
    if '0' in target:
        incl_trials[trials['signed_contrast'] != 0] = False  # Only include 0% contrast
#     if ('prior' in target) and ('stim' in target):
#         incl_trials[trials['signed_contrast'] != 0] = False  # Only include 0% contrast
    incl_trials[trials['reaction_times'] < min_rt] = False  # Exclude trials with fast rt
    
    # Brandon added 211014 convert format
    trials_incl_trials = {}
    for k in trials.keys():
        if trials[k] is not None:
            trials_incl_trials[k] = trials[k][incl_trials]
        else:
            trials_incl_trials[k] = None
    trials_incl_trials
    return trials_incl_trials

def get_target_from_model(TARGET, SAVE_PATH, subject,
                          stimuli_arr, actions_arr, stim_sides_arr, session_uuids,
                          REMOVE_OLD_FIT):
    '''
    Brandon got from Guido 2021 and built on it
    '''
    # Get maximum number of trials across sessions
    max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

    # Pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials
    stimuli = np.array([np.concatenate((stimuli_arr[k], np.zeros(max_len-len(stimuli_arr[k]))))
                        for k in range(len(stimuli_arr))])
    actions = np.array([np.concatenate((actions_arr[k], np.zeros(max_len-len(actions_arr[k]))))
                        for k in range(len(actions_arr))])
    stim_side = np.array([np.concatenate((stim_sides_arr[k],
                                          np.zeros(max_len-len(stim_sides_arr[k]))))
                          for k in range(len(stim_sides_arr))])

    # define function to retrieve targets and model parameters, params
#     if 'stimside' in TARGET:
    model = exp_stimside(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                             session_uuids, subject, actions, stimuli, stim_side)
#     elif 'prevaction' in TARGET: 
#         model = exp_prev_action(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
#                                 session_uuids, subject, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    params = model.get_parameters(parameter_type='posterior_mean')

    if 'prior' in TARGET:
        target = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side,
                                      parameter_type='posterior_mean', verbose=False)['prior']

    elif 'prederr' in TARGET:
        target = model.compute_signal(signal='prediction_error', act=actions, stim=stimuli,
                                      side=stim_side, verbose=False,
                                      parameter_type='posterior_mean')['prediction_error']
        
    target = np.squeeze(np.array(target))
        
    return target, params

def getatlassummary(xyz):
    atlascent = (np.mean(xyz[0]),np.mean(xyz[1]),np.mean(xyz[2]))
    dxyz = xyz[0] - atlascent[0],xyz[1] - atlascent[1],xyz[2] - atlascent[2]
    atlasdists = np.sqrt(dxyz[0]**2 + dxyz[1]**2 + dxyz[2]**2)
    atlasradi = np.mean(atlasdists)
    return atlascent, atlasradi