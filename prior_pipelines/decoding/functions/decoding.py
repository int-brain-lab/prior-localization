import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import linear_model as sklm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from tqdm import tqdm
from behavior_models.models.utils import format_data as format_data_mut
from behavior_models.models.utils import format_input as format_input_mut
from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
from sklearn.utils.class_weight import compute_sample_weight

from ibllib.atlas import BrainRegions

from prior_pipelines.decoding.functions.process_inputs import select_ephys_regions
from prior_pipelines.decoding.functions.process_inputs import get_bery_reg_wfi
from prior_pipelines.decoding.functions.process_inputs import (
    select_widefield_imaging_regions,
)
from prior_pipelines.decoding.functions.neurometric import compute_neurometric_prior
from prior_pipelines.decoding.functions.process_inputs import preprocess_ephys
from prior_pipelines.decoding.functions.process_inputs import preprocess_widefield_imaging
from prior_pipelines.decoding.functions.process_targets import compute_beh_target

from prior_pipelines.decoding.functions.utils import compute_mask
from prior_pipelines.decoding.functions.utils import save_region_results
from prior_pipelines.decoding.functions.utils import get_save_path
from prior_pipelines.decoding.functions.nulldistributions import (
    generate_null_distribution_session,
)
from prior_pipelines.decoding.functions.process_targets import check_bhv_fit_exists
from prior_pipelines.decoding.functions.process_targets import optimal_Bayesian
from prior_pipelines.decoding.functions.neurometric import get_neurometric_parameters
from prior_pipelines.decoding.functions.utils import derivative

from prior_pipelines.decoding.functions.process_motors import (
    preprocess_motors,
    compute_motor_prediction,
)


def fit_eid(neural_dict, trials_df, metadata, pseudo_ids=[-1], **kwargs):
    """High-level function to decode a given target variable from brain regions for a single eid.

    Parameters
    ----------
    neural_dict : dict
        keys: 'spk_times', 'spk_clu', 'clu_regions', 'clu_qc', 'clu_df'
    trials_df : dict
        columns: 'choice', 'feedback', 'pLeft', 'firstMovement_times', 'stimOn_times',
        'feedback_times'
    metadata : dict
        'eid', 'eid_train', 'subject', 'probes'
    pseudo_ids : array-like
        whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered.
        if pseudo_id>0, a pseudo session is used. cannot be 0.
    kwargs
        target : str
            single-bin targets: 'pLeft' | 'signcont' | 'choice' | 'feedback'
            multi-bin targets: 'wheel-vel' | 'wheel-speed' | 'pupil' | '[l/r]-paw-pos'
                | '[l/r]-paw-vel' | '[l/r]-paw-speed' | '[l/r]-whisker-me'
        align_time : str
            event in trial on which to align intervals
            'firstMovement_times' | 'stimOn_times' | 'feedback_times'
        time_window : tuple
            (window_start, window_end), relative to align_time
        binsize : float
            size of bins in seconds for multi-bin decoding
        n_bins_lag : int
            number of lagged bins to use for predictors for multi-bin decoding
        estimator : sklearn.linear_model object
            sklm.Lasso | sklm.Ridge | sklm.LinearRegression | sklm.LogisticRegression
        hyperparam_grid : dict
            regularization values to search over
        n_runs : int
            number of independent runs performed. this was added after variability was observed
            across runs.
        shuffle : bool
            True for interleaved cross-validation, False for contiguous blocks
        min_units : int
            minimum units per region to use for decoding
        qc_criteria : float
            fraction between 0 and 1 that describes the number of qc tests that need to be passed
            in order to use each unit. 0 means all units are used; 1 means a unit has to pass
            every qc test in order to be used
        min_behav_trials : int
            minimum number of trials (after filtering) that must be present to proceed with fits
        min_rt : float
            minimum reaction time per trial; can be used to filter out trials with negative
            reaction times
        no_unbias : bool
            True to remove unbiased trials; False to keep
        neural_dtype : str
            'ephys' | 'widefield'
        today : str
            date string for specifying filenames
        output_path : str
            absolute path where decoding fits are saved
        add_to_saving_path : str
            additional string to append to filenames
    """

    print(f"Working on eid : %s" % metadata["eid"])
    filenames = []  # this will contain paths to saved decoding results for this eid

    if 0 in pseudo_ids:
        raise ValueError(
            "pseudo id can be -1 (actual session) or strictly greater than 0 (pseudo session)"
        )

    if not np.all(np.sort(pseudo_ids) == pseudo_ids):
        raise ValueError("pseudo_ids must be sorted")

    # if you want to train the model on one session or all sessions
    if "eids_train" not in metadata.keys():
        metadata["eids_train"] = [metadata["eid"]]
    
    # train model if not trained already
    if kwargs["model"] != optimal_Bayesian and kwargs["model"] is not None:
        side, stim, act, _ = format_data_mut(trials_df)
        stimuli, actions, stim_side = format_input_mut([stim], [act], [side])
        behmodel = kwargs["model"](
            kwargs["behfit_path"],
            np.array(metadata["eids_train"]),
            metadata["subject"],
            actions,
            stimuli,
            stim_side,
            single_zeta=True,
        )
        istrained, _ = check_bhv_fit_exists(
            metadata["subject"],
            kwargs["model"],
            metadata["eids_train"],
            kwargs["behfit_path"],
            modeldispatcher=kwargs["modeldispatcher"],
            single_zeta=True,
        )
        if not istrained:
            behmodel.load_or_train(remove_old=False)

    if kwargs["target"] in ["pLeft", "signcont", "strengthcont", "choice", "feedback"]:
        target_vals_list = compute_beh_target(trials_df, metadata, **kwargs)
        mask_target = np.ones(len(target_vals_list), dtype=bool)
    else:
        raise NotImplementedError('this case is not implemented')
    
    mask = compute_mask(trials_df, **kwargs) & mask_target

    if sum(mask) <= kwargs["min_behav_trials"]:
        msg = "session contains %i trials, below the threshold of %i" % (
            sum(mask),
            kwargs["min_behav_trials"],
        )
        logging.exception(msg)
        return filenames

    # select brain regions from beryl atlas to loop over
    brainreg = BrainRegions()
    beryl_reg = (
        brainreg.acronym2acronym(neural_dict["clu_regions"], mapping="Beryl")
        if kwargs["neural_dtype"] == "ephys"
        else get_bery_reg_wfi(neural_dict, **kwargs)
    )

    if isinstance(kwargs["single_region"], bool):
        regions = (
            [[k] for k in np.unique(beryl_reg)]
            if kwargs["single_region"]
            else [np.unique(beryl_reg)]
        )
    else:
        if kwargs["single_region"] == "Custom":
            regions = [["VISp"], ["MOs"]]
        elif kwargs["single_region"] == "Widefield":
            regions = [
                ["ACAd"],
                ["AUDd"],
                ["AUDp"],
                ["AUDpo"],
                ["AUDv"],
                ["FRP"],
                ["MOB"],
                ["MOp"],
                ["MOs"],
                ["PL"],
                ["RSPagl"],
                ["RSPd"],
                ["RSPv"],
                ["SSp-bfd"],
                ["SSp-ll"],
                ["SSp-m"],
                ["SSp-n"],
                ["SSp-tr"],
                ["SSp-ul"],
                ["SSp-un"],
                ["SSs"],
                ["TEa"],
                ["VISa"],
                ["VISal"],
                ["VISam"],
                ["VISl"],
                ["VISli"],
                ["VISp"],
                ["VISpl"],
                ["VISpm"],
                ["VISpor"],
                ["VISrl"],
            ]
        else:
            regions = (
                [[kwargs["single_region"]]]
                if isinstance(kwargs["single_region"], str)
                else [kwargs["single_region"]]
            )

        if np.all([reg not in np.unique(beryl_reg) for reg in regions]):
            return filenames

    for region in tqdm(regions, desc="Region: ", leave=False):

        if kwargs["neural_dtype"] == "ephys":
            reg_clu_ids = select_ephys_regions(neural_dict, beryl_reg, region, **kwargs)
        elif kwargs["neural_dtype"] == "widefield":
            reg_mask = select_widefield_imaging_regions(neural_dict, region, **kwargs)
        else:
            raise NotImplementedError

        if kwargs["neural_dtype"] == "ephys" and len(reg_clu_ids) < kwargs["min_units"]:
            print(region, "below min units threshold :", len(reg_clu_ids))
            continue

        if kwargs["neural_dtype"] == "ephys":
            msub_binned = preprocess_ephys(
                reg_clu_ids, neural_dict, trials_df, **kwargs
            )
            n_units = len(reg_clu_ids)
        elif kwargs["neural_dtype"] == "widefield":
            msub_binned = preprocess_widefield_imaging(neural_dict, reg_mask, **kwargs)
            n_units = np.sum(reg_mask)
        else:
            raise NotImplementedError

        ##### motor signal regressors #####

        if kwargs.get("motor_regressors", None):
            print("motor regressors")
            motor_binned = preprocess_motors(
                metadata["eid"], kwargs
            )  # size (nb_trials,nb_motor_regressors) => one bin per trial

            if kwargs["motor_regressors_only"]:
                msub_binned = motor_binned
            else:
                msub_binned = np.concatenate([msub_binned, motor_binned], axis=2)

        ##################################

        # make feature matrix
        Xs = msub_binned
        if msub_binned[0].shape[0] != 1:
            raise AssertionError('Decoding is only supported when the target is one dimensional')

        fit_results = []
        for pseudo_id in pseudo_ids:

            # create pseudo session when necessary
            if pseudo_id > 0:
                if kwargs['set_seed_for_DEBUG']:
                    np.random.seed(pseudo_id)
                controlsess_df = generate_null_distribution_session(
                    trials_df, metadata, **kwargs
                )
                controltarget_vals_list = compute_beh_target(
                    controlsess_df, metadata, **kwargs
                )
                save_predictions = kwargs.get(
                    "save_predictions_pseudo", kwargs["save_predictions"]
                )
            else:
                save_predictions = kwargs["save_predictions"]

            if kwargs["compute_neurometric"]:  # compute prior for neurometric curve
                trialsdf_neurometric = (
                    trials_df.reset_index()
                    if (pseudo_id == -1)
                    else controlsess_df.reset_index()
                )
                trialsdf_neurometric = compute_neurometric_prior(
                    trialsdf_neurometric, metadata, **kwargs
                )

            ### derivative of target signal before mask application ###
            if kwargs["decode_derivative"]:
                if pseudo_id == -1:
                    target_vals_list = derivative(target_vals_list)
                else:
                    controltarget_vals_list = derivative(controltarget_vals_list)

            ### replace target signal by residual of motor prediction ###
            if kwargs["motor_residual"]:
                if pseudo_id == -1:
                    motor_prediction = compute_motor_prediction(
                        metadata["eid"], target_vals_list, kwargs
                    )
                    target_vals_list = target_vals_list - motor_prediction
                else:
                    motor_prediction = compute_motor_prediction(
                        metadata["eid"], controltarget_vals_list, kwargs
                    )
                    controltarget_vals_list = controltarget_vals_list - motor_prediction

            y_decoding = (
                [target_vals_list[m] for m in np.squeeze(np.where(mask))]
                if pseudo_id == -1
                else [controltarget_vals_list[m] for m in np.squeeze(np.where(mask))]
            )

            # run decoders
            for i_run in range(kwargs["nb_runs"]):

                if kwargs["set_seed_for_DEBUG"]:
                    rng_seed = i_run
                else:
                    rng_seed = None

                fit_result = decode_cv(
                    ys=y_decoding,
                    Xs=[Xs[m] for m in np.squeeze(np.where(mask))],
                    estimator=kwargs["estimator"],
                    estimator_kwargs=kwargs["estimator_kwargs"],
                    hyperparam_grid=kwargs["hyperparam_grid"],
                    save_binned=kwargs["save_binned"],
                    save_predictions=save_predictions,
                    shuffle=kwargs["shuffle"],
                    balanced_weight=kwargs["balanced_weight"],
                    rng_seed=rng_seed,
                    use_cv_sklearn_method=kwargs[
                        "use_native_sklearn_for_hyperparameter_estimation"
                    ],
                )
                fit_result["mask"] = mask if save_predictions else None
                fit_result["df"] = trials_df if pseudo_id == -1 else controlsess_df
                fit_result["pseudo_id"] = pseudo_id
                fit_result["run_id"] = i_run

                # compute neurometric curves
                if kwargs["compute_neurometric"]:
                    (
                        fit_result["full_neurometric"],
                        fit_result["fold_neurometric"],
                    ) = get_neurometric_parameters(
                        fit_result,
                        trialsdf=trialsdf_neurometric[mask.values]
                        .reset_index()
                        .drop("index", axis=1),
                        compute_on_each_fold=kwargs["compute_on_each_fold"],
                    )
                else:
                    fit_result["full_neurometric"] = None
                    fit_result["fold_neurometric"] = None
                fit_results.append(fit_result)

        # save out decoding results
        if kwargs["neural_dtype"] == "ephys":
            probe = metadata["probe_name"]
        elif kwargs["neural_dtype"] == "widefield":
            probe = metadata["hemispheres"]

        save_path = get_save_path(
            pseudo_ids,
            metadata["subject"],
            metadata["eid"],
            kwargs["neural_dtype"],
            probe=probe,
            region=str(np.squeeze(region)) if kwargs["single_region"] else "allRegions",
            output_path=kwargs["neuralfit_path"],
            time_window=kwargs["time_window"],
            today=kwargs["date"],
            target=kwargs["target"],
            add_to_saving_path=kwargs["add_to_saving_path"],
        )

        if kwargs['run_integration_test']:
            save_path = save_path.parent.joinpath(save_path.name.split('.pkl')[0] + '_to_be_tested.pkl')

        filename = save_region_results(
            fit_result=fit_results,
            pseudo_id=pseudo_ids,
            subject=metadata["subject"],
            eid=metadata["eid"],
            probe=probe,
            region=region,
            n_units=n_units,
            save_path=save_path,
        )
        

        filenames.append(filename)

    return filenames


def decode_cv(
    ys,
    Xs,
    estimator,
    estimator_kwargs,
    balanced_weight=False,
    hyperparam_grid=None,
    test_prop=0.2,
    n_folds=5,
    save_binned=False,
    save_predictions=True,
    verbose=False,
    shuffle=True,
    outer_cv=True,
    rng_seed=None,
    use_cv_sklearn_method=False,
):
    """Regresses binned neural activity against a target, using a provided sklearn estimator.

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
