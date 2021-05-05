"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import brainbox.modeling.design_matrix as dm
import brainbox.modeling.linear as lm
import brainbox.modeling.utils as mut
from synth import _generate_pseudo_blocks
import brainbox.io.one as bbone
from datetime import date
import pickle
import os
from tqdm import tqdm
from copy import deepcopy

offline = True
one = one.ONE()


def fit_session(session_id, kernlen, nbases,
                t_before=1., t_after=0.6, max_len=2., probe_idx=0,
                contnorm=5., binwidth=0.02, abswheel=False, no_50perc=False, num_pseudosess=100,
                target_regressor='pLeft'):
    if not abswheel:
        signwheel = True
    else:
        signwheel = False
    trdf = bbone.load_trials_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after,
                                wheel_binsize=binwidth, ret_abswheel=abswheel,
                                ret_wheel=signwheel)
    probestr = 'probe0' + str(probe_idx)
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                aligned=True)
    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    clu_regions = clusters[probestr].acronym

    trdf['pLeft_last'] = pd.Series(np.roll(trdf['probabilityLeft'], 1),
                                   index=trdf.index)[:-1]
    trdf = trdf.iloc[1:-1]
    trdf['adj_contrastLeft'] = np.tanh(
        contnorm * trdf['contrastLeft']) / np.tanh(contnorm)
    trdf['adj_contrastRight'] = np.tanh(
        contnorm * trdf['contrastRight']) / np.tanh(contnorm)

    if no_50perc:
        trdf = trdf[trdf.probabilityLeft != 0.5]

    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
                'pLeft_last': 'value',
                'feedbackType': 'value',
                'feedback_times': 'timing',
                'contrastLeft': 'value',
                'adj_contrastLeft': 'value',
                'contrastRight': 'value',
                'adj_contrastRight': 'value',
                'goCue_times': 'timing',
                'stimOn_times': 'timing',
                'trial_start': 'timing',
                'trial_end': 'timing',
                'bias': 'value',
                'bias_next': 'value',
                'wheel_velocity': 'continuous'}

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.pLeft_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.zeros(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times + 0.1)
        currtr_end = design.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.pLeft_last
        zerovec[currtr_end:] = row.probabilityLeft
        return zerovec

    design = dm.DesignMatrix(trdf, vartypes, binwidth=binwidth)
    stepbounds = [design.binf(t_before - 0.6), design.binf(t_before - 0.1)]

    cosbases_long = mut.full_rcos(kernlen, nbases, design.binf)
    cosbases_short = mut.full_rcos(0.4, nbases, design.binf)
    design.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastLeft',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastRight',
                                desc='Kernel conditioned on R stimulus onset')
    design.add_covariate_timing('correct', 'feedback_times', cosbases_long,
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')
    design.add_covariate_raw('pLeft', stepfunc_prestim,
                             desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr', stepfunc_poststim,
                             desc='Step function on post-stimulus prior')
    design.add_covariate('wheel', trdf['wheel_velocity'], cosbases_short, -0.4)
    design.compile_design_matrix()

    _, s, v = np.linalg.svd(design[:, design.covar['wheel']['dmcol_idx']], full_matrices=False)
    variances = s**2 / (s**2).sum()
    n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
    wheelcols = design[:, design.covar['wheel']['dmcol_idx']]
    reduced = wheelcols @ v[:n_keep].T
    bases_reduced = cosbases_short @ v[:n_keep].T
    keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar['wheel']['dmcol_idx'])
    basedm = design[:, keepcols]
    design.dm = np.hstack([basedm, reduced])
    design.covar['wheel']['dmcol_idx'] = design.covar['wheel']['dmcol_idx'][:n_keep]
    design.covar['wheel']['bases'] = bases_reduced

    print(np.linalg.cond(design.dm))
    trialinds = np.array([(tr, np.searchsorted(design.trlabels.flat, tr))
                          for tr in design.trialsdf.index])
    tmparr = np.roll(trialinds[:, 1], -1)
    tmparr[-1] = design.dm.shape[0]
    trialinds = np.hstack((trialinds, tmparr.reshape(-1, 1)))

    nglm = lm.LinearGLM(design, spk_times, spk_clu, estimator=RidgeCV(cv=3))
    glm_template = deepcopy(nglm)
    nglm.fit(printcond=False)
    realscores = nglm.score()
    nglm.clu_regions = clu_regions

    scoreslist = []
    weightslist = []
    for _ in tqdm(range(num_pseudosess), desc='Pseudo block iteration num', leave=False):
        newblocks = _generate_pseudo_blocks(len(design.trialsdf))
        probmap = {1: 0.8, 0: 0.2}
        newprobs = [probmap[i] for i in newblocks]
        tmp_df = design.trialsdf.copy()
        tmp_df['probabilityLeft'] = newprobs
        tmp_df['pLeft_last'] = pd.Series(np.roll(tmp_df['probabilityLeft'], 1),
                                         index=tmp_df.index)
        tmpglm = deepcopy(glm_template)
        pl_idx = design.covar['pLeft']['dmcol_idx']
        plt_idx = design.covar['pLeft_tr']['dmcol_idx']
        for tr, start, end in trialinds:
            if target_regressor == 'pLeft':
                tmpglm.design.dm[start:end, pl_idx][tmpglm.design.dm[start:end, pl_idx]
                                                    > 0] = tmp_df.pLeft_last[tr]
            else:
                pl_old = trdf.probabilityLeft[tr]
                pll_old = trdf.pLeft_last[tr]
                tmpglm.design.dm[start:end, plt_idx][tmpglm.design.dm[start:end,
                                                     plt_idx] == pl_old] = tmp_df.probabilityLeft[tr]
                tmpglm.design.dm[start:end, plt_idx][tmpglm.design.dm[start:end,
                                                     plt_idx] == pll_old] = tmp_df.pLeft_last[tr]
        tmpglm.fit(printcond=False)
        weightslist.append((tmpglm.coefs, tmpglm.intercepts))
        with np.errstate(all='ignore'):
            scoreslist.append(tmpglm.score())
    return nglm, realscores, scoreslist, weightslist


def get_bwm_ins_alyx(one):
    """
    Return insertions that match criteria :
    - project code
    - session QC not critical (TODO may need to add probe insertion QC)
    - at least 1 alignment
    - behavior pass
    :return:
    ins: dict containing the full details on insertion as per the alyx rest query
    ins_id: list of insertions eids
    sess_id: list of (unique) sessions eids
    """
    ins = one.alyx.rest('insertions', 'list',
                        provenance='Ephys aligned histology track',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               '~json__qc,CRITICAL,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    sessions = {}
    for item in ins:
        s_eid = item['session_info']['id']
        if s_eid not in sessions:
            sessions[s_eid] = []
        sessions[s_eid].append(item['id'])
    return sessions


if __name__ == "__main__":
    from glob import glob
    import traceback

    # currdate = str(date.today())
    currdate = '2021-05-04'
    sessions = get_bwm_ins_alyx(one)

    savepath = '/home/berk/Documents/fits/'

    abswheel = True
    kernlen = 0.6
    nbases = 10
    stepwise = True
    wholetrial_step = False
    no_50perc = True
    method = 'pure'
    blocking = False
    prior_estimate = None
    binwidth = 0.02
    n_pseudo = 100
    target = 'pLeft'

    print(f'Fitting {len(sessions)} sessions...')

    for sessid in sessions:
        info = one.get_details(sessid)

        nickname = info['subject']
        sessdate = info['start_time'][:10]
        print(f'\nWorking on {nickname} from {sessdate}\n')
        for i in range(len(sessions[sessid])):
            probe = i
            filename = savepath + \
                f'{nickname}/{sessdate}_session_{currdate}_probe{probe}_fit.p'
            if not os.path.exists(savepath + nickname):
                os.mkdir(savepath + nickname)
            subpaths = [n for n in glob(
                os.path.abspath(filename)) if os.path.isfile(n)]
            if len(subpaths) != 0:
                print(
                    f'Skipped {nickname}, {sessdate}, probe{probe}: already fit.')
            if len(subpaths) == 0:
                print(sessid)
                if not offline:
                    _ = one.load(sessid, download_only=True)
                try:
                    outtup = fit_session(sessid, kernlen, nbases, t_before=0.7, probe_idx=probe,
                                         abswheel=abswheel, no_50perc=no_50perc,
                                         num_pseudosess=n_pseudo, target_regressor=target)
                    nglm, realscores, scoreslist, weightslist = outtup
                except Exception as err:
                    tb = traceback.format_exc()
                    print(
                        f'Subject {nickname} on {sessdate} failed:\n', type(err), err)
                    print(tb)
                    continue

                outdict = {'sessinfo': {'eid': sessid, 'nickname': nickname, 'sessdate': sessdate},
                           'kernlen': kernlen, 'nbases': nbases, 'method': method,
                           'binwidth': binwidth, 'realscores': realscores, 'scores': scoreslist,
                           'weightslist': weightslist, 'fitobj': nglm}
                today = str(date.today())
                subjfilepath = os.path.abspath(filename)
                fw = open(subjfilepath, 'wb')
                pickle.dump(outdict, fw)
                fw.close()
