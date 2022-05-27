import copy
import numpy as np
import pytest
import sklearn.linear_model as sklm

from braindelphi.decoding.functions.decoding import fit_eid
from braindelphi.decoding.functions.decoding import decode_cv


def test_decode_cv():

    n_trials = 50
    n_bins_per_trial = 1
    n_clusters = 4
    n_folds = 4

    # basic regression run with Ridge estimator
    ys = [np.random.randn(1,) for _ in range(n_trials)]
    Xs = [np.random.randn(n_bins_per_trial, n_clusters) for _ in range(n_trials)]
    estimator = sklm.Ridge
    results1 = decode_cv(
        ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
        target_distribution=None, bin_size_kde=1.0,
        hyperparam_grid={'alpha': np.array([0.1, 1.0])},
        n_folds=n_folds, shuffle=True,
        save_binned=True, save_predictions=True, rng_seed=0)
    assert isinstance(results1, dict)
    assert 'scores_test_full' in results1
    assert isinstance(results1['scores_test_full'], np.float64)
    assert 'scores_train' in results1
    assert len(results1['scores_train']) == n_folds
    assert 'scores_test' in results1
    assert len(results1['scores_test']) == n_folds
    assert 'Rsquared_test_full' in results1
    assert isinstance(results1['Rsquared_test_full'], np.float64)
    assert 'acc_test_full' not in results1
    assert 'balanced_acc_test_full' not in results1
    assert 'weights' in results1
    assert len(results1['weights']) == n_folds
    assert 'intercepts' in results1
    assert len(results1['intercepts']) == n_folds
    assert 'target' in results1
    assert len(results1['target']) == n_trials
    assert 'predictions_test' in results1
    assert len(results1['predictions_test']) == n_trials
    assert 'regressors' in results1
    assert len(results1['regressors']) == n_trials
    assert 'idxes_test' in results1
    assert len(results1['idxes_test']) == n_folds
    assert 'idxes_train' in results1
    assert len(results1['idxes_train']) == n_folds
    assert 'best_params' in results1
    assert len(results1['best_params']) == n_folds
    assert 'n_folds' in results1
    assert isinstance(results1['n_folds'], int)

    # results are reproducible
    results2 = decode_cv(
        ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
        target_distribution=None, bin_size_kde=1.0,
        hyperparam_grid={'alpha': np.array([0.1, 1.0])},
        n_folds=n_folds,
        save_binned=True, save_predictions=True, rng_seed=0)  # <- same rng seed
    assert np.allclose(np.concatenate(
        results1['predictions_test']), np.concatenate(results2['predictions_test']))

    # results change w/ rng seed
    results3 = decode_cv(
        ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
        target_distribution=None, bin_size_kde=1.0,
        hyperparam_grid={'alpha': np.array([0.1, 1.0])},
        n_folds=n_folds,
        save_binned=True, save_predictions=True, rng_seed=1)  # <- different rng seed
    assert ~np.allclose(
        np.concatenate(results1['predictions_test']),
        np.concatenate(results3['predictions_test'])
    )

    # shuffle=True lead to shuffled trials
    for fold in results1['idxes_test']:  # results1 comes from above
        assert ~np.all(np.diff(fold) == 1)

    # shuffle=False leads to unshuffled trials
    results = decode_cv(
        ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
        target_distribution=None, bin_size_kde=1.0,
        hyperparam_grid={'alpha': np.array([0.1, 1.0])},
        n_folds=n_folds, shuffle=False,
        save_binned=True, save_predictions=True)
    for fold in results['idxes_test']:
        assert np.all(np.diff(fold) == 1)

    # basic regression run with Lasso
    estimator = sklm.Lasso
    results = decode_cv(
        ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
        target_distribution=None, bin_size_kde=1.0,
        hyperparam_grid={'alpha': np.array([0.1, 1.0])},
        n_folds=n_folds,
        save_binned=True, save_predictions=True)
    assert isinstance(results, dict)

    # don't use CV estimator for ridge
    estimator = sklm.RidgeCV
    with pytest.raises(NotImplementedError):
        decode_cv(
            ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
            target_distribution=None, bin_size_kde=1.0,
            hyperparam_grid={'alpha': np.array([0.1, 1.0])},
            n_folds=n_folds,
            save_binned=True, save_predictions=True)

    # don't use CV estimator for lasso
    estimator = sklm.LassoCV
    with pytest.raises(NotImplementedError):
        decode_cv(
            ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
            target_distribution=None, bin_size_kde=1.0,
            hyperparam_grid={'alpha': np.array([0.1, 1.0])},
            n_folds=n_folds,
            save_binned=True, save_predictions=True)

    # logistic regression
    estimator = sklm.LogisticRegression
    ys = [np.random.randint(0, 2, (1,)) for _ in range(n_trials)]
    results = decode_cv(
        ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
        target_distribution=None, bin_size_kde=1.0,
        hyperparam_grid={'C': np.array([0.1, 1.0])},
        n_folds=n_folds,
        save_binned=True, save_predictions=True)
    assert isinstance(results, dict)

    # don't use CV estimator for logistic regression
    estimator = sklm.LogisticRegressionCV
    with pytest.raises(NotImplementedError):
        decode_cv(
            ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
            target_distribution=None, bin_size_kde=1.0,
            hyperparam_grid={'alpha': np.array([0.1, 1.0])},
            n_folds=n_folds,
            save_binned=True, save_predictions=True)

    # normalize input/output works
    ys = [np.random.randn(1,) for _ in range(n_trials)]
    Xs = [np.random.randn(n_bins_per_trial, n_clusters) for _ in range(n_trials)]
    estimator = sklm.Ridge
    results = decode_cv(
        ys=ys, Xs=Xs, estimator=estimator, estimator_kwargs={}, use_openturns=False,
        target_distribution=None, bin_size_kde=1.0,
        hyperparam_grid={'alpha': np.array([0.1, 1.0])},
        n_folds=n_folds,
        save_binned=True, save_predictions=True,
        normalize_input=True, normalize_output=True)
    assert isinstance(results, dict)


def test_fit_eid(data_dict, tmp_path):

    from braindelphi.decoding.settings import fit_metadata

    # toy metadata
    metadata = {
        'eid': '1234',
        # 'eids_train': None,
        'subject': 'MRW_000', 'probe': 'probe00', 'merge_probes': True
    }

    # update paths in metadata so that pytest will clean them up automatically
    fit_metadata['output_path'] = tmp_path
    fit_metadata['min_behav_trials'] = 0

    # this will be more of an integration test to make sure the full fit_eid function runs on real
    # data

    # fit on oracle prior
    fit_metadata['target'] = 'pLeft'  # prior
    fit_metadata['model'] = None  # oracle prior
    fit_eid(
        neural_dict=data_dict, trials_df=data_dict['trialsdf'], metadata=copy.copy(metadata),
        dlc_dict=None, pseudo_ids=[-1], **fit_metadata)

    # fit on oracle prior pseudo-session
    fit_metadata['use_imposter_session'] = False
    fit_eid(
        neural_dict=data_dict, trials_df=data_dict['trialsdf'], metadata=copy.copy(metadata),
        dlc_dict=None, pseudo_ids=[1], **fit_metadata)

    # TODO: use imposter session
    # TODO: decode other targets: single-bin
    # TODO: decode other targets: multi-bin
