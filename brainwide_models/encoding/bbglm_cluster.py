"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
import os
import sys
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from brainbox.task.closed_loop import generate_pseudo_session
from bbglm_sessfit import fit_stepwise, generate_design
from params import BEH_MOD_PATH, GLM_FIT_PATH, GLM_CACHE
sys.path.append(Path(__file__).joinpath('decoding'))
from decoding.functions.utils import compute_target, query_sessions  # noqa
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember


def filter_nan(trialsdf):
    target_cols = ['stimOn_times', 'feedback_times', 'firstMovement_times']
    mask = ~np.any(np.isnan(trialsdf[target_cols]), axis=1)
    return trialsdf[mask]


def get_cached_regressors(fpath):
    with open(fpath, 'rb') as fo:
        d = pickle.load(fo)
    return d['trialsdf'], d['spk_times'], d['spk_clu'], d['clu_regions'], d['clu_qc']


def save_pseudo(subject, session_id, fitout, params):
    # TODO: Make this work
    raise NotImplementedError
    # sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    # fn = sesspath.joinpath()
    # subjfilepath = os.path.abspath(filename)
    # fw = open(subjfilepath, 'wb')
    # pickle.dump(outdict, fw)
    # fw.close()
    # return True


def save_stepwise(subject, session_id, fitout, params, probes, input_fn, clu_reg, clu_qc, fitdate):
    sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    fn = sesspath.joinpath(f'{fitdate}_stepwise_regression.pkl')
    outdict = {
        'params': params,
        'probes': probes,
        'model_input_fn': input_fn,
        'clu_regions': clu_reg,
        'clu_qc': clu_qc,
    }
    outdict.update(fitout)
    with open(fn, 'wb') as fw:
        pickle.dump(outdict, fw)
    return fn


def compute_deltas(scores):
    outdf = pd.DataFrame(np.zeros_like(scores), index=scores.index, columns=scores.columns)
    for i in scores.columns:  # Change this for diff num covs
        if i >= 1:
            diff = scores[i] - scores[i - 1]
        else:
            diff = scores[i]
        outdf [i] = diff
    return outdf


def colrename(cname, suffix):
    return str(cname + 1) + 'cov' + suffix


def _create_sub_sess_path(parent, subject, session):
    subpath = Path(parent).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session)
    if not sesspath.exists():
        os.mkdir(sesspath)
    return sesspath


def remap(ids, source='Allen', dest='Beryl', output='acronym', br=BrainRegions()):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']


def get_id(acronym, brainregions=BrainRegions()):
    return brainregions.id[np.argwhere(brainregions.acronym == acronym)[0, 0]]


def get_name(acronym, brainregions=BrainRegions()):
    if acronym == 'void':
        return acronym
    reg_idxs = np.argwhere(brainregions.acronym == acronym).flat
    return brainregions.name[reg_idxs[0]]


def label_cerebellum(acronym, brainregions=BrainRegions()):
    regid = brainregions.id[np.argwhere(brainregions.acronym == acronym).flat][0]
    ancestors = brainregions.ancestors(regid)
    if 'Cerebellum' in ancestors.name or 'Medulla' in ancestors.name:
        return True
    else:
        return False


if __name__ == "__main__":
    import brainbox.modeling.utils as mut
    import brainbox.modeling.linear as lm
    import sklearn.linear_model as skl
    from sklearn.model_selection import GridSearchCV

    # Model parameters
    def tmp_binf(t):
        return np.ceil(t / params['binwidth']).astype(int)

    params = {
        'binwidth': 0.02,
        'iti_prior': [-0.4, -0.1],
        'fmove_offset': -0.2,
        'wheel_offset': -0.3,
        'contnorm': 5.,
        'reduce_wheel_dim': True,
        'dataset_fn': '2022-01-19_dataset_metadata.pkl',
        'model': lm.LinearGLM,
        'alpha_grid': {'alpha': np.logspace(-2, 1.5, 50)},
        'contiguous': False,
        # 'n_pseudo': 100,
    }

    params['bases'] = {
        'stim': mut.raised_cosine(0.4, 5, tmp_binf),
        'feedback': mut.raised_cosine(0.4, 5, tmp_binf),
        'wheel': mut.raised_cosine(0.3, 3, tmp_binf),
        'fmove': mut.raised_cosine(0.2, 3, tmp_binf),
    }
    params['estimator'] = GridSearchCV(skl.Ridge(), params['alpha_grid'])

    currdate = str(date.today())
    # currdate = '2021-05-04'

    parpath = Path(GLM_FIT_PATH).joinpath(f'{currdate}_glm_fit_pars.pkl')
    datapath = Path(GLM_CACHE).joinpath(params['dataset_fn'])
    with open(parpath, 'wb') as fw:
        pickle.dump(params, fw)
    print("Parameters file located at:", parpath)
    print("Dataset file used:", datapath)

    sessions = query_sessions('resolved-behavior').set_index(['subject', 'eid'])
    with open(datapath, 'rb') as fo:
        dataset = pickle.load(fo)
    dataset_params = dataset['params']
    dataset_fns = dataset['dataset_filenames']


    ###########################################################################
    # Before running below code, workers must process sessions and save files #
    ###########################################################################
    n_cov = 8  # Modify if you change the model!

    filenames = []
    for subj in os.listdir(Path(GLM_FIT_PATH)):
        subjdir = Path(GLM_FIT_PATH).joinpath(subj)
        if not os.path.isdir(subjdir):
            continue
        for sess in os.listdir(subjdir):
            sessdir = subjdir.joinpath(sess)
            for file in os.listdir(sessdir):
                filepath = sessdir.joinpath(file)
                if os.path.isfile(filepath) and filepath.match(f'*{currdate}*'):
                    filenames.append(filepath)

    # Process files after fitting
    sessdfs = []
    for fitname in filenames:
        with open(fitname, 'rb') as fo:
            tmpfile = pickle.load(fo)
        folds = []
        for i in range(len(tmpfile['scores'])):
            tmp_sc = tmpfile['scores'][i].rename(columns=lambda c: colrename(c, '_score'))
            tmp_seq = tmpfile['sequences'][i].rename(columns=lambda c: colrename(c, '_name'))
            tmp_diff = compute_deltas(tmpfile['scores'][i])
            tmp_diff.rename(columns=lambda c: colrename(c, '_diff'), inplace=True)
            tmpdf = tmp_sc.join(tmp_seq).join(tmp_diff)
            tmpdf['eid'] = fitname.parts[-2]
            tmpdf['acronym'] = tmpfile['clu_regions'][tmpdf.index]
            tmpdf['qc_label'] = tmpfile['clu_qc']['label'][tmpdf.index]
            tmpdf['fold'] = i
            tmpdf.index.set_names(['clu_id'], inplace=True)
            folds.append(tmpdf.reset_index())
        sess_master = pd.concat(folds)
        sessdfs.append(sess_master)
    masterscores = pd.concat(sessdfs)

    for i in range(1, n_cov + 1):  # Change this for diff num covs
        if i >= 2:
            diff = masterscores[str(i) + 'cov_score'] - masterscores[str(i - 1) + 'cov_score']
        else:
            diff = masterscores[str(i) + 'cov_score']
    masterscores[str(i) + 'cov_diff'] = diff

    br = BrainRegions()
    grpby = masterscores.groupby('acronym')
    masterscores['reg_id'] = grpby.acronym.transform(lambda g: get_id(g.unique()[0], br))
    masterscores['beryl_acr'] = grpby.reg_id.transform(lambda g: remap(g, br=br))
    masterscores['cerebellum'] = grpby.acronym.transform(lambda g: label_cerebellum(g.unique()[0],
                                                                                    br))
    masterscores['region'] = masterscores['beryl_acr']
    masterscores['name'] = grpby.region.transform(lambda g: get_name(g.unique()[0], br))

    masterscores.set_index(['eid', 'acronym', 'clu_id', 'fold'], inplace=True)

    design_example = generate_design()
    outdict = {
        'fit_params': params,
        'dataset': dataset,
        'fit_results': masterscores,
        'fit_files': filenames,
        'design_example': design_example
    }
    with open(Path(GLM_FIT_PATH).joinpath(f'{currdate}_glm_fit.pkl'), 'wb') as fw:
        pickle.dump(outdict, fw)
