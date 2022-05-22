

def regress_target(tvec,
                   binned,
                   estimatorObject,
                   estimator_kwargs,
                   use_openturns,
                   target_distribution,
                   bin_size_kde,
                   balanced_continuous_target=True,
                   hyperparam_grid=None,
                   test_prop=0.2,
                   nFolds=5,
                   save_binned=False,
                   verbose=False,
                   shuffle=True,
                   outer_cv=True,
                   balanced_weight=False,
                   normalize_input=False,
                   normalize_output=False):
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
    scores_test, scores_train, weights, intercepts = [], [], [], []
    predictions, predictions_test, idxes_test, idxes_train, best_params = [], [], [], [], []
    predictions_test_to_save = []

    # train / test split
    # Split the dataset in two equal parts
    # when shuffle=False, the method will take the end of the dataset to create the test set
    indices = np.arange(len(tvec))
    if outer_cv:
        outer_kfold = KFold(n_splits=nFolds, shuffle=shuffle).split(indices)
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])

    # scoring function
    scoring_f = balanced_accuracy_score if (estimatorObject
                                            == sklm.LogisticRegression) else r2_score

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if isinstance(estimatorObject, LinearModelCV):
        if hyperparam_grid is not None:
            raise TypeError('If using a CV estimator hyperparam_grid will not be respected;'
                            ' set to None')
        estimatorObject.cv = nFolds  # Overwrite user spec to make sure nFolds is used
        clf = estimatorObject
        raise NotImplemented('the code does not support a CV-type estimator for the moment.')
    else:
        for train_index, test_index in outer_kfold:
            X_train, X_test = binned[train_index], binned[test_index]
            y_train, y_test = tvec[train_index], tvec[test_index]

            idx_inner = np.arange(len(X_train))
            inner_kfold = KFold(n_splits=nFolds, shuffle=shuffle).split(idx_inner)

            key = list(hyperparam_grid.keys())[0]
            r2s = np.zeros([nFolds, len(hyperparam_grid[key])])
            for ifold, (train_inner, test_inner) in enumerate(inner_kfold):
                X_train_inner, X_test_inner = X_train[train_inner], X_train[test_inner]
                y_train_inner, y_test_inner = y_train[train_inner], y_train[test_inner]

                # normalization when necessary
                mean_X_train_inner = X_train_inner.mean(axis=0) if normalize_input else 0
                X_train_inner = X_train_inner - mean_X_train_inner
                X_test_inner = X_test_inner - mean_X_train_inner
                mean_y_train_inner = y_train_inner.mean(axis=0) if normalize_output else 0
                y_train_inner = y_train_inner - mean_y_train_inner

                for i_alpha, alpha in enumerate(hyperparam_grid[key]):
                    estimator = estimatorObject(**{**estimator_kwargs, key: alpha})
                    if balanced_weight:
                        estimator.fit(X_train_inner,
                                      y_train_inner,
                                      sample_weight=balanced_weighting(
                                          vec=y_train_inner,
                                          continuous=balanced_continuous_target,
                                          use_openturns=use_openturns,
                                          bin_size_kde=bin_size_kde,
                                          target_distribution=target_distribution))
                    else:
                        estimator.fit(X_train_inner, y_train_inner)
                    pred_test_inner = estimator.predict(X_test_inner) + mean_y_train_inner

                    r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

            r2s_avg = r2s.mean(axis=0)
            best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
            clf = estimatorObject(**{**estimator_kwargs, key: best_alpha})

            # normalization when necessary
            mean_X_train = X_train.mean(axis=0) if normalize_input else 0
            X_train = X_train - mean_X_train
            mean_y_train = y_train.mean(axis=0) if normalize_output else 0
            y_train = y_train - mean_y_train

            if balanced_weight:
                clf.fit(X_train,
                        y_train,
                        sample_weight=balanced_weighting(vec=y_train,
                                                         continuous=balanced_continuous_target,
                                                         use_openturns=use_openturns,
                                                         bin_size_kde=bin_size_kde,
                                                         target_distribution=target_distribution))
            else:
                clf.fit(X_train, y_train)

            # compute R2 on the train data
            y_pred_train = clf.predict(X_train)
            scores_train.append(scoring_f(y_train + mean_y_train, y_pred_train + mean_y_train))

            # compute R2 on held-out data
            y_true, prediction = y_test, clf.predict(binned - mean_X_train) + mean_y_train
            scores_test.append(scoring_f(y_true, prediction[test_index]))

            # save the raw prediction in the case of linear and the predicted proba when working with logit
            prediction_to_save = (prediction if not (estimatorObject == sklm.LogisticRegression)
                                  else clf.predict_proba(binned - mean_X_train)[:, 0] +
                                  mean_y_train)

            # prediction, target, idxes_test, idxes_train
            predictions.append(prediction)
            predictions_test.append(prediction[test_index])
            predictions_test_to_save.append(prediction_to_save[test_index])
            idxes_test.append(test_index)
            idxes_train.append(train_index)
            weights.append(clf.coef_)
            if clf.fit_intercept:
                intercepts.append(clf.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    full_test_predictions_to_save = np.zeros(len(tvec))
    full_test_prediction = np.zeros(len(tvec))
    for k in range(nFolds):
        full_test_prediction[idxes_test[k]] = predictions_test[k]
        full_test_predictions_to_save[idxes_test[k]] = predictions_test_to_save[k]

    outdict = dict()
    outdict['score_test_full'] = scoring_f(tvec, full_test_prediction)
    outdict['scores_train'] = scores_train
    outdict['scores_test'] = scores_test
    outdict['Rsquared_test_full'] = r2_score(tvec, full_test_prediction)
    if estimatorObject == sklm.LogisticRegression:
        outdict['acc_test_full'] = accuracy_score(tvec, full_test_prediction)
        outdict['balanced_acc_test_full'] = balanced_accuracy_score(tvec, full_test_prediction)
    outdict['weights'] = weights
    outdict['intercepts'] = intercepts
    outdict['target'] = tvec
    outdict['predictions_test'] = np.array(full_test_predictions_to_save)
    outdict['idxes_test'] = idxes_test
    outdict['idxes_train'] = idxes_train
    outdict['best_params'] = best_params
    outdict['nFolds'] = nFolds
    if save_binned:
        outdict['regressors'] = binned
    if hasattr(clf, 'classes_'):
        outdict['classes_'] = clf.classes_

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
            tscore_fold = list(
                np.round(clf.cv_results_['split{}_test_score'.format(int(i_fold))], 3))
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
