# Third party libraries
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm

# IBL libraries
import brainbox.modeling.utils as mut


def fit(design, spk_t, spk_clu, binwidth, model, estimator, n_folds=5, contiguous=False, **kwargs):
    trials_idx = design.trialsdf.index
    nglm = model(design, spk_t, spk_clu, binwidth=binwidth, estimator=estimator)
    splitter = KFold(n_folds, shuffle=not contiguous)
    scores, weights, intercepts, alphas, splits = [], [], [], [], []
    for test, train in splitter.split(trials_idx):
        nglm.fit(train_idx=train, printcond=False)
        if isinstance(estimator, GridSearchCV):
            alphas.append(estimator.best_params_['alpha'])
        elif isinstance(estimator, RegressorMixin):
            alphas.append(estimator.get_params()['alpha'])
        else:
            raise TypeError('Estimator must be a sklearn linear regression instance')
        intercepts.append(nglm.intercepts)
        weights.append(nglm.combine_weights())
        scores.append(nglm.score(testinds=test))
        splits.append({'test': test, 'train': train})
    outdict = {
        'scores': scores,
        'weights': weights,
        'intercepts': intercepts,
        'alphas': alphas,
        'splits': splits
    }
    return outdict


def fit_stepwise(design,
                 spk_t,
                 spk_clu,
                 binwidth,
                 model,
                 estimator,
                 n_folds=5,
                 contiguous=False,
                 **kwargs):
    trials_idx = design.trialsdf.index
    nglm = model(design, spk_t, spk_clu, binwidth=binwidth, estimator=estimator)
    splitter = KFold(n_folds, shuffle=not contiguous)
    sequences, scores, splits = [], [], []
    for test, train in tqdm(splitter.split(trials_idx), desc='Fold', leave=False):
        nglm.traininds = train
        sfs = mut.SequentialSelector(nglm)
        sfs.fit()
        sequences.append(sfs.sequences_)
        scores.append(sfs.scores_)
        # TODO: Extract per-submodel alpha values
        splits.append({'test': test, 'train': train})
    outdict = {'scores': scores, 'sequences': sequences, 'splits': splits}
    return outdict
