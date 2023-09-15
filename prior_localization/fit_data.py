import logging
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import linear_model as sklm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
from sklearn.utils.class_weight import compute_sample_weight

from brainbox.io.one import SessionLoader

from prior_localization.prepare_data import (prepare_ephys, prepare_behavior, prepare_motor, prepare_pupil,
                                             prepare_widefield, prepare_widefield_old)
from prior_localization.functions.neurometric import get_neurometric_parameters
from prior_localization.functions.utils import (create_neural_path, check_inputs, check_config, compute_mask,
                                                subtract_motor_residuals)

# Set up logger
logger = logging.getLogger('prior_localization')
# Load settings
with open(Path(__file__).parent.joinpath('settings.yml'), "r") as settings_yml:
    settings = yaml.safe_load(settings_yml)
# Load and check configuration file
config = check_config()


def fit_session_ephys(
        one, session_id, subject, probe_name, model='optBay', pseudo_ids=None, target='pLeft',
        align_event='stimOn_times', time_window=(-0.6, -0.1), output_dir=None, regions='single_regions',
        min_trials=150, motor_residuals=False, stage_only=False, integration_test=False
):
    """
    Fits a single session for ephys data.
    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    probe_name: str or list of str
     Probe name(s)
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    pseudo_ids: list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    time_window: tuple of float
     Time window in which neural activity is considered, relative to align_event, default is (-0.6, -0.1)
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. If None, use working directory
    regions: str or list of str
     Specify the level of granularity at which decoding is done. Options are 'single_regions' (every region that is
     passed by the probe is decoded separately), 'all_regions' (all regions passed are decoded together), any key
     into the 'regions_defaults' dictionary, specified in config.yml, or a single region name or list of region names
    min_trials: int
     Minimum number of valid trials for session to be decoded, default is 150
    motor_residuals: bool
     Whether ot compute the motor residual before performing neural decoding. This argument is used to study embodiment
     corresponding to figure 2f, default is False
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding
    integration_test: bool
     If true set random seeds for integration testing. Do not use this when running actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)
    if motor_residuals and model != 'optBay':
        raise ValueError('Motor residuals can only be computed for optBay model')

    # Load trials data and compute mask
    sl = SessionLoader(one, session_id)
    sl.load_trials()
    trials_mask = compute_mask(sl.trials, align_event=align_event, min_rt=0.08, max_rt=None, n_trials_crop_end=0)
    if sum(trials_mask) <= min_trials:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {min_trials}.")

    # Compute or load behavior targets
    all_trials, all_targets, trials_mask, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, output_dir=output_dir,
        model=model, target=target,  compute_neurometrics=config['compute_neurometrics'],
        integration_test=integration_test)

    # Remove the motor residuals from the targets if indicated
    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, trials_mask = subtract_motor_residuals(motor_signals, all_targets, trials_mask)

    # Prepare ephys data
    intervals = np.vstack([sl.trials[align_event] + time_window[0], sl.trials[align_event] + time_window[1]]).T
    data_epoch, actual_regions = prepare_ephys(
        one, session_id, probe_name, regions, intervals, qc=config['unit_qc'], min_units=config['min_units'],
        stage_only=stage_only
    )

    # Fix the probe name (mainly for saving)
    probe_name = 'merged_probes' if isinstance(probe_name, list) else probe_name

    # If we are only staging data, we are done here
    if stage_only:
        return

    # Otherwise fit per region
    filenames = []
    for data_region, region in zip(data_epoch, actual_regions):
        # Create a string for saving the file
        region_str = regions if (regions == 'all_regions') or (
            regions in config['region_defaults'].keys()) else '_'.join(region)

        # Fit
        fit_results = fit_target(data_region[trials_mask], [t[trials_mask] for t in all_targets], all_trials,
                                 all_neurometrics, pseudo_ids, integration_test=integration_test)

        # Add the mask to fit results
        for fit_result in fit_results:
            fit_result['mask'] = trials_mask if config['save_predictions'] else None

        # Create output paths and save
        filename = create_neural_path(output_dir, settings['date'], 'ephys', subject, session_id, probe_name,
                                      region_str, target, time_window, pseudo_ids, settings['add_to_path'])
        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "probe": probe_name,
            "region": region,
            "N_units": data_region.shape[1],
        }
        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)
        filenames.append(filename)
    return filenames


def fit_session_widefield(
        one, session_id, subject, hemisphere=("left", "right"), model='optBay', pseudo_ids=None, target='pLeft',
        align_event='stimOn_times', frame_window=(-2, -2), output_dir=None, regions='widefield', min_trials=150,
        stage_only=False, integration_test=False, old_data=False):

    """
    Fit a single session for widefield data.

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    hemisphere: str or tuple of str
        Which hemisphere(s) to decode from {'left', 'right', ('left', 'right')}
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    pseudo_ids: list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    frame_window: tuple of int
     Window in which neural activity is considered, in frames relative to align_event, default is (-2, -2) i.e. only a
     single frame is considered
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. If None, use working directory
    regions: str or list of str
     Specify the level of granularity at which decoding is done. Options are 'single_regions' (every region that is
     passed by the probe is decoded separately), 'all_regions' (all regions passed are decoded together), any key
     into the 'regions_defaults' dictionary, specified in config.yml, or a single region name or list of region names
    min_trials: int
     Minimum number of valid trials for session to be decoded, default is 150
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding
    integration_test: bool
     If true set random seeds for integration testing. Do not use this when running actual decoding
    old_data: False or str
     Only used for sanity check, if false, use updated way of loading data from ONE. If str it should be a path
     to local copies of the previously used version of the data.

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)

    # Load trials data
    sl = SessionLoader(one, session_id)
    sl.load_trials()
    trials_mask = compute_mask(sl.trials, align_event=align_event, min_rt=0.08, max_rt=None, n_trials_crop_end=1)
    if sum(trials_mask) <= min_trials:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {min_trials}.")

    # Compute or load behavior targets
    all_trials, all_targets, trials_mask, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, output_dir=output_dir,
        model=model, target=target,  compute_neurometrics=config['compute_neurometrics'],
        integration_test=integration_test)

    # Prepare widefield data
    if old_data is False:
        data_epoch, actual_regions = prepare_widefield(
            one, session_id, regions=regions, align_times=sl.trials[align_event].values, frame_window=frame_window,
            hemisphere=hemisphere, stage_only=stage_only
        )
    else:
        data_epoch, actual_regions = prepare_widefield_old(old_data, hemisphere=hemisphere, regions=regions,
                                                           align_event=align_event, frame_window=frame_window)

    # Hemishpere name (mainly for saving)
    hemi_name = 'both_hemispheres' if isinstance(hemisphere, tuple) or isinstance(hemisphere, list) else hemisphere

    # If we are only staging data, we are done here
    if stage_only:
        return

    # Otherwise, fit data per region
    filenames = []
    for data_region, region in zip(data_epoch, actual_regions):
        region_str = regions if (regions == 'all_regions') else '_'.join(region)
        fit_results = fit_target(data_region[trials_mask], [t[trials_mask] for t in all_targets], all_trials,
                                 all_neurometrics, pseudo_ids, integration_test=integration_test)

        # Add the mask to fit results
        for fit_result in fit_results:
            fit_result['mask'] = trials_mask if config['save_predictions'] else None

        # Create output paths and save
        filename = create_neural_path(output_dir, settings['date'], 'widefield', subject, session_id, hemi_name,
                                      region_str, target, frame_window, pseudo_ids, config['add_to_path'])
        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "probe": hemisphere,
            "region": region,
            "N_units": data_region.shape[1],
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)
        filenames.append(filename)
    return filenames


def fit_session_pupil(
        one, session_id, subject, model='optBay', pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        time_window=(-0.6, -0.1), output_dir=None, min_trials=150, stage_only=False, integration_test=False
):
    """
    Fit pupil tracking data to behavior (instead of neural activity)

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    pseudo_ids: list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    time_window: tuple of float
     Time window in which pupil movement is considered, relative to align_event, default is (-0.6, -0.1)
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. If None, use working directory
    min_trials: int
     Minimum number of valid trials for session to be decoded, default is 150
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding
    integration_test: bool
     If true set random seeds for integration testing. Do not use this when running actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """
    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)

    # Load trials data
    sl = SessionLoader(one, session_id)
    sl.load_trials()
    trials_mask = compute_mask(sl.trials, align_event=align_event, min_rt=0.08, max_rt=None, n_trials_crop_end=0)
    if sum(trials_mask) <= min_trials:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {min_trials}.")

    # Compute or load behavior targets
    all_trials, all_targets, trials_mask, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, output_dir=output_dir,
        model=model, target=target,  compute_neurometrics=config['compute_neurometrics'],
        integration_test=integration_test)

    # Load the pupil data
    pupil_data = prepare_pupil(one, session_id=session_id, time_window=time_window, align_event=align_event)

    if stage_only:
        return

    # For trials where there was no pupil data recording (start/end), add these to the trials_mask
    trials_mask = trials_mask & ~np.any(np.isnan(pupil_data), axis=1)

    # Fit
    fit_results = fit_target(pupil_data[trials_mask], [t[trials_mask] for t in all_targets], all_trials,
                             all_neurometrics, pseudo_ids, integration_test=integration_test)

    # Create output paths and save
    filename = create_neural_path(
        output_path=output_dir, date=settings['date'], neural_dtype='ephys', subject=subject,
        session_id=session_id, probe='', region_str='pupil', target=target, time_window=time_window,
        pseudo_ids=pseudo_ids, add_to_path=settings['add_to_path']
    )

    outdict = {
        "fit": fit_results,
        "subject": subject,
        "eid": session_id,
        "probe": None,
    }
    with open(filename, "wb") as fw:
        pickle.dump(outdict, fw)

    return filename


def fit_session_motor(
        one, session_id, subject, model='optBay', pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        time_window=(-0.6, -0.1), output_dir=None, stage_only=False, min_trials=150, integration_test=False
):
    """
    Fit movement tracking data to behavior (instead of neural actvity)

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    pseudo_ids: list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    time_window: tuple of float
     Time window in which movement is considered, relative to align_event, default is (-0.6, -0.1)
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. If None, use working directory
    min_trials: int
     Minimum number of valid trials for session to be decoded, default is 150
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding
    integration_test: bool
     If true set random seeds for integration testing. Do not use this when running actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)

    # Load trials data
    sl = SessionLoader(one, session_id)
    sl.load_trials()
    trials_mask = compute_mask(sl.trials, align_event=align_event, min_rt=0.08, max_rt=None, n_trials_crop_end=0)
    if sum(trials_mask) <= min_trials:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {min_trials}.")

    # Compute or load behavior targets
    all_trials, all_targets, trials_mask, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, output_dir=output_dir,
        model=model, target=target,  compute_neurometrics=config['compute_neurometrics'],
        integration_test=integration_test)

    # Load the motor data
    motor_data = prepare_motor(one, session_id=session_id, time_window=time_window, align_event=align_event)

    if stage_only:
        return

    # For trials where there was no motor data (start/end), add these to the trials_mask
    trials_mask = trials_mask & ~np.any(np.isnan(motor_data), axis=1)

    # Fit
    fit_results = fit_target(motor_data[trials_mask], [t[trials_mask] for t in all_targets], all_trials,
                             all_neurometrics, pseudo_ids, integration_test=integration_test)

    # Create output paths and save
    filename = create_neural_path(
        output_path=output_dir, date=settings['date'], neural_dtype='ephys', subject=subject,
        session_id=session_id, probe='', region_str='motor', target=target, time_window=time_window,
        pseudo_ids=pseudo_ids, add_to_path=settings['add_to_path']
    )

    outdict = {
        "fit": fit_results,
        "subject": subject,
        "eid": session_id,
        "probe": None,
    }
    with open(filename, "wb") as fw:
        pickle.dump(outdict, fw)

    return filename


def fit_target(data_to_fit, all_targets, all_trials, all_neurometrics=None, pseudo_ids=None, integration_test=False):
    """
    Fits data (neural, motor, etc) to behavior targets.

    Parameters
    ----------
    data_to_fit : list of np.ndarray
        List of neural or other data, each element is a (n_trials, n_units) array with the averaged neural activity
    all_targets : list of np.ndarray
        List of behavior targets, each element is a (n_trials,) array with the behavior targets for one (pseudo)session
    all_trials : list of pd.DataFrames
        List of trial information, each element is a pd.DataFrame with the trial information for one (pseudo)session
    all_neurometrics : list of pd.DataFrames or None
        List of neurometrics, each element is a pd.DataFrame with the neurometrics for one (pseudo)session.
        If None, don't compute neurometrics. Default is None
    pseudo_ids : list of int or None
        List of pseudo session ids, -1 indicates the actual session. If None, run only on actual session.
        Default is None.
    integration_test : bool
        Whether to run in integration test mode with fixed random seeds. Default is False.
    """

    # Loop over (pseudo) sessions and then over runs
    if pseudo_ids is None:
        pseudo_ids = [-1]
    if not all_neurometrics:
        all_neurometrics = [None] * len(all_targets)
    fit_results = []
    for targets, trials, neurometrics, pseudo_id in zip(all_targets, all_trials, all_neurometrics, pseudo_ids):
        # run decoders
        for i_run in range(config['n_runs']):
            rng_seed = i_run if integration_test else None
            fit_result = decode_cv(
                ys=targets,
                Xs=data_to_fit,
                estimator=config['estimator'],
                estimator_kwargs=config['estimator_kwargs'],
                hyperparam_grid=config['hparam_grid'],
                save_binned=False,
                save_predictions=config['save_predictions'],
                shuffle=config['shuffle'],
                balanced_weight=config['balanced_weighting'],
                rng_seed=rng_seed,
                use_cv_sklearn_method=config['use_native_sklearn_for_hyperparam_estimation'],
            )

            fit_result["trials_df"] = trials
            fit_result["pseudo_id"] = pseudo_id
            fit_result["run_id"] = i_run

            if neurometrics:
                fit_result["full_neurometric"], fit_result["fold_neurometric"] = get_neurometric_parameters(
                    fit_result, trialsdf=neurometrics, compute_on_each_fold=config['compute_neuro_on_each_fold']
                )
            else:
                fit_result["full_neurometric"] = None
                fit_result["fold_neurometric"] = None

            fit_results.append(fit_result)

    return fit_results


def decode_cv(ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
              n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
              rng_seed=None, use_cv_sklearn_method=False):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    Parameters
    ----------
    ys : list of arrays or np.ndarray or pandas.Series
        targets; if list, each entry is an array of targets for one trial. if 1D numpy array, each
        entry is treated as a single scalar for one trial. if pd.Series, trial number is the index
        and teh value is the target.
    Xs : list of arrays or np.ndarray
        predictors; if list, each entry is an array of neural activity for one trial. if 2D numpy
        array, each row is treated as a single vector of ativity for one trial, i.e. the array is
        of shape (n_trials, n_neurons)
    estimator : sklearn.linear_model object
        estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
    estimator_kwargs : dict
        additional arguments for sklearn estimator
    balanced_weight : bool
        balanced weighting to target
    hyperparam_grid : dict
        key indicates hyperparameter to grid search over, and value is an array of nodes on the
        grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
        Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
    test_prop : float
        proportion of data to hold out as the test set after running hyperparameter tuning; only
        used if `outer_cv=False`
    n_folds : int
        Number of folds for cross-validation during hyperparameter tuning; only used if
        `outer_cv=True`
    save_binned : bool
        True to put the regressors Xs into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    save_predictions : bool
        True to put the model predictions into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    shuffle : bool
        True for interleaved cross-validation, False for contiguous blocks
    outer_cv: bool
        Perform outer cross validation such that the testing spans the entire dataset
    rng_seed : int
        control data splits
    verbose : bool
        Whether you want to hear about the function's life, how things are going, and what the
        neighbor down the street said to it the other day.

    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Decoding intercept
            - Per-trial target values (copy of tvec)
            - Per-trial predictions from model
            - Input regressors (optional, see Xs argument)

    """

    # transform target data into standard format: list of np.ndarrays
    if isinstance(ys, np.ndarray):
        # input is single numpy array
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, list) and ys[0].shape == ():
        # input is list of float instead of list of np.ndarrays
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, pd.Series):
        # input is a pandas Series
        ys = ys.to_numpy()
        ys = [np.array([y]) for y in ys]

    # transform neural data into standard format: list of np.ndarrays
    if isinstance(Xs, np.ndarray):
        Xs = [x[None, :] for x in Xs]

    # initialize containers to save outputs
    n_trials = len(Xs)
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [
        None for _ in range(n_trials)
    ]  # different for logistic regression

    # split the dataset in two parts, train and test
    # when shuffle=False, the method will take the end of the dataset to create the test set
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)
    if outer_cv:
        outer_kfold = KFold(
            n_splits=n_folds if not use_cv_sklearn_method else 50, shuffle=shuffle
        ).split(indices)
    else:
        outer_kfold = iter(
            [train_test_split(indices, test_size=test_prop, shuffle=shuffle)]
        )

    # scoring function; use R2 for linear regression, accuracy for logistic regression
    scoring_f = (
        balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
    )

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if (
        estimator == sklm.RidgeCV
        or estimator == sklm.LassoCV
        or estimator == sklm.LogisticRegressionCV
    ):
        raise NotImplementedError(
            "the code does not support a CV-type estimator for the moment."
        )
    else:
        # loop over outer folds
        for train_idxs_outer, test_idxs_outer in outer_kfold:
            # outer fold data split
            # X_train = np.vstack([Xs[i] for i in train_idxs])
            # y_train = np.concatenate([ys[i] for i in train_idxs], axis=0)
            # X_test = np.vstack([Xs[i] for i in test_idxs])
            # y_test = np.concatenate([ys[i] for i in test_idxs], axis=0)
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]  # TODO: make this more robust

            if not use_cv_sklearn_method:
                # now loop over inner folds
                idx_inner = np.arange(len(X_train))
                inner_kfold = KFold(n_splits=n_folds, shuffle=shuffle).split(idx_inner)

                r2s = np.zeros([n_folds, len(hyperparam_grid[key])])
                inner_predictions = (
                    np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                )
                inner_targets = (
                    np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                )
                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(
                    inner_kfold
                ):

                    # inner fold data split
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate(
                        [y_train[i] for i in train_idxs_inner], axis=0
                    )
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate(
                        [y_train[i] for i in test_idxs_inner], axis=0
                    )

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                        # compute weight for each training sample if requested
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None

                        # initialize model
                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        # fit model
                        model_inner.fit(
                            X_train_inner, y_train_inner, sample_weight=sample_weight
                        )
                        # evaluate model
                        pred_test_inner = (
                            model_inner.predict(X_test_inner)
                        )
                        inner_predictions[test_idxs_inner, i_alpha] = pred_test_inner
                        inner_targets[test_idxs_inner, i_alpha] = y_test_inner
                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))
                r2s_avg = np.array(
                    [
                        scoring_f(inner_targets[:, _k], inner_predictions[:, _k])
                        for _k in range(len(hyperparam_grid[key]))
                    ]
                )
                # select model with best hyperparameter value evaluated on inner-fold test data;
                # refit/evaluate on all inner-fold data
                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # compute weight for each training sample if requested
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                # initialize model
                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                # fit model
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
            else:
                if (
                    estimator not in [Ridge, Lasso]
                ):
                    raise NotImplementedError("This case is not implemented")
                model = (
                    RidgeCV(alphas=hyperparam_grid[key])
                    if estimator == Ridge
                    else LassoCV(alphas=hyperparam_grid[key])
                )
                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            # evalute model on train data
            y_pred_train = model.predict(X_train_array)
            scores_train.append(
                scoring_f(y_train_array, y_pred_train)
            )

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)
            y_pred = model.predict(np.vstack(X_test))
            if isinstance(estimator, sklm.LogisticRegression):
                y_pred_probs = (
                    model.predict_proba(np.vstack(X_test))[:, 0]
                )
            else:
                y_pred_probs = None
            scores_test.append(scoring_f(y_true, y_pred))

            # save the raw prediction in the case of linear and the predicted probabilities when
            # working with logitistic regression
            for i_fold, i_global in enumerate(test_idxs_outer):
                # we already computed these estimates, take from above
                predictions[i_global] = np.array([y_pred[i_fold]])
                if isinstance(estimator, sklm.LogisticRegression):
                    predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                else:
                    predictions_to_save[i_global] = np.array([y_pred[i_fold]])

            # save out other data of interest
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    ys_true_full = np.concatenate(ys, axis=0)
    ys_pred_full = np.concatenate(predictions, axis=0)
    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)
    if estimator == sklm.LogisticRegression:
        outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
        outdict["balanced_acc_test_full"] = balanced_accuracy_score(
            ys_true_full, ys_pred_full
        )
    outdict["weights"] = weights if save_predictions else None
    outdict["intercepts"] = intercepts if save_predictions else None
    outdict["target"] = ys
    outdict["predictions_test"] = predictions_to_save
    outdict["regressors"] = Xs if save_binned else None
    outdict["idxes_test"] = idxes_test if save_predictions else None
    outdict["idxes_train"] = idxes_train if save_predictions else None
    outdict["best_params"] = best_params if save_predictions else None
    outdict["n_folds"] = n_folds
    if hasattr(model, "classes_"):
        outdict["classes_"] = model.classes_

    # logging
    if verbose:
        # verbose output
        if outer_cv:
            print("Performance is only described for last outer fold \n")
        print(
            "Possible regularization parameters over {} validation sets:".format(
                n_folds
            )
        )
        print("{}: {}".format(list(hyperparam_grid.keys())[0], hyperparam_grid))
        print("\nBest parameters found over {} validation sets:".format(n_folds))
        print(model.best_params_)
        print("\nAverage scores over {} validation sets:".format(n_folds))
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, model.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n", "Detailed scores on {} validation sets:".format(n_folds))
        for i_fold in range(n_folds):
            tscore_fold = list(
                np.round(model.cv_results_["split{}_test_score".format(int(i_fold))], 3)
            )
            print("perf on fold {}: {}".format(int(i_fold), tscore_fold))

        print("\n", "Detailed classification report:", "\n")
        print("The model is trained on the full (train + validation) set.")

    return outdict
