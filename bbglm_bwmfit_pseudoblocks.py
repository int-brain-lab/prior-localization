"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
from brainbox.modeling import glm, glm_linear
from brainbox.population.population import _generate_pseudo_blocks
import brainbox.io.one as bbone
import alf.io as aio
from prior_funcs import fit_sess_psytrack
from datetime import date
import pickle
import os
from tqdm import tqdm
from copy import deepcopy

offline = True
one = one.ONE()

ephys_cache = {}


def fit_session(session_id, kernlen, nbases,
                t_before=1., t_after=0.6, prior_estimate='psytrack', max_len=2., probe_idx=0,
                method='minimize', contnorm=5., binwidth=0.02, wholetrial_step=False,
                stepwise=False, blocktrain=False, abswheel=False, no_50perc=False,
                num_pseudosess=100):
    if not abswheel:
        signwheel = True
    else:
        signwheel = False
    # if offline:
    #     path = one.path_from_eid(session_id)
    #     if not aio.exists(path.joinpath('alf', 'probe00'), object='spikes'):
    #         raise ValueError('Session not downloaded locally. Aborting.')
    trialsdf = bbone.load_trials_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after,
                                    wheel_binsize=binwidth, ret_abswheel=abswheel,
                                    ret_wheel=signwheel)
    if prior_estimate == 'psytrack':
        print('Fitting psytrack esimates...')
        wts, stds = fit_sess_psytrack(session_id, maxlength=max_len, as_df=True)
        wts['bias'] = wts['bias'] - np.mean(wts['bias'])
        fitinfo = pd.concat((trialsdf, wts['bias']), axis=1)
        bias_next = np.roll(fitinfo['bias'], -1)
        bias_next = pd.Series(bias_next, index=fitinfo['bias'].index)[:-1]
        fitinfo['bias_next'] = bias_next
    elif prior_estimate is None:
        fitinfo = trialsdf.copy()
    else:
        raise NotImplementedError('Only psytrack currently available')
    # spk_times = one.load(session_id, dataset_types=['spikes.times'], offline=offline)[probe_idx]
    # spk_clu = one.load(session_id, dataset_types=['spikes.clusters'], offline=offline)[probe_idx]

    # A bit of messy loading to get spike times, clusters, and cluster brain regions.
    # This is the way it is because loading with regions takes forever. The weird for loop
    # ensures that we don't waste memory storing unnecessary and large arrays.
    probestr = 'probe0' + str(probe_idx)
    if session_id not in ephys_cache:
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                    aligned=True)
        for probe in spikes:
            null_keys_spk = []
            for key in spikes[probe]:
                if key not in ('times', 'clusters'):
                    null_keys_spk.append(key)
            for key in null_keys_spk:
                _ = spikes[probe].pop(key)
            null_keys_clu = []
            for key in clusters[probe]:
                if key not in ('acronym', 'atlas_id'):
                    null_keys_clu.append(key)
            for key in null_keys_clu:
                _ = clusters[probe].pop(key)
        ephys_cache[session_id] = (spikes, clusters)
    else:
        spikes, clusters = ephys_cache[session_id]
    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    clu_regions = clusters[probestr].acronym

    fitinfo['pLeft_last'] = pd.Series(np.roll(fitinfo['probabilityLeft'], 1),
                                      index=fitinfo.index)[:-1]
    fitinfo = fitinfo.iloc[1:-1]
    fitinfo['adj_contrastLeft'] = np.tanh(contnorm * fitinfo['contrastLeft']) / np.tanh(contnorm)
    fitinfo['adj_contrastRight'] = np.tanh(contnorm * fitinfo['contrastRight']) / np.tanh(contnorm)

    if no_50perc:
        fitinfo = fitinfo[fitinfo.probabilityLeft != 0.5]

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
    if t_before < 0.7:
        raise ValueError('t_before needs to be 0.7 or greater in order to do -0.1 to -0.7 step'
                         ' function on pLeft')

    def stepfunc(row):
        currvec = np.ones(nglm.binf(row.duration)) * row.pLeft_last
        # nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.feedback_times)) *\
        #     row.probabilityLeft
        return currvec

    def stepfunc_prestim(row):
        stepvec = np.zeros(nglm.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.pLeft_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.ones(nglm.binf(row.duration))
        currtr_start = nglm.binf(row.stimOn_times + 0.1)
        currtr_end = nglm.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.pLeft_last
        zerovec[currtr_end:] = row.probabilityLeft
        return zerovec

    nglm = glm_linear.LinearGLM(fitinfo, spk_times, spk_clu, vartypes, binwidth=binwidth,
                                blocktrain=blocktrain, subset=stepwise)
    nglm.clu_regions = clu_regions
    stepbounds = [nglm.binf(t_before - 0.6), nglm.binf(t_before - 0.1)]

    cosbases_long = glm.full_rcos(kernlen, nbases, nglm.binf)
    cosbases_short = glm.full_rcos(0.4, nbases, nglm.binf)
    nglm.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
                              cond=lambda tr: np.isfinite(tr.contrastLeft),
                              deltaval='adj_contrastLeft',
                              desc='Kernel conditioned on L stimulus onset')
    nglm.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
                              cond=lambda tr: np.isfinite(tr.contrastRight),
                              deltaval='adj_contrastRight',
                              desc='Kernel conditioned on R stimulus onset')
    nglm.add_covariate_timing('correct', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == 1,
                              desc='Kernel conditioned on correct feedback')
    nglm.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == -1,
                              desc='Kernel conditioned on incorrect feedback')
    if prior_estimate is None and wholetrial_step:
        nglm.add_covariate_raw('pLeft', stepfunc, desc='Step function on prior estimate')
    elif prior_estimate is None and not wholetrial_step:
        nglm.add_covariate_raw('pLeft', stepfunc_prestim,
                               desc='Step function on prior estimate')
        nglm.add_covariate_raw('pLeft_tr', stepfunc_poststim,
                               desc='Step function on post-stimulus prior')
    nglm.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
    nglm.compile_design_matrix()

    _, s, v = np.linalg.svd(nglm.dm[:, nglm.covar['wheel']['dmcol_idx']], full_matrices=False)
    variances = s**2 / (s**2).sum()
    n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
    wheelcols = nglm.dm[:, nglm.covar['wheel']['dmcol_idx']]
    reduced = wheelcols @ v[:n_keep].T
    bases_reduced = cosbases_short @ v[:n_keep].T
    keepcols = ~np.isin(np.arange(nglm.dm.shape[1]), nglm.covar['wheel']['dmcol_idx'])
    basedm = nglm.dm[:, keepcols]
    nglm.dm = np.hstack([basedm, reduced])
    nglm.covar['wheel']['dmcol_idx'] = nglm.covar['wheel']['dmcol_idx'][:n_keep]
    nglm.covar['wheel']['bases'] = bases_reduced

    glm_template = deepcopy(nglm)

    print(np.linalg.cond(nglm.dm))
    trialinds = np.array([(tr, np.searchsorted(nglm.trlabels.flat, tr))
                          for tr in nglm.trialsdf.index])
    tmparr = np.roll(trialinds[:, 1], -1)
    tmparr[-1] = nglm.dm.shape[0]
    trialinds = np.hstack((trialinds, tmparr.reshape(-1, 1)))

    nglm.fit(method=method, printcond=False)
    realscores = nglm.score()

    scoreslist = []
    weightslist = []
    for i in tqdm(range(num_pseudosess), desc='Pseudo block iteration num', leave=False):
        newblocks = _generate_pseudo_blocks(len(glm_template.trialsdf))
        probmap = {1: 0.8, 0: 0.2}
        newprobs = [probmap[i] for i in newblocks]
        tmp_df = glm_template.trialsdf.copy()
        tmp_df['probabilityLeft'] = newprobs
        tmp_df['pLeft_last'] = pd.Series(np.roll(tmp_df['probabilityLeft'], 1),
                                         index=tmp_df.index)
        tmpglm = deepcopy(glm_template)
        pl_idx = tmpglm.covar['pLeft']['dmcol_idx']
        plt_idx = tmpglm.covar['pLeft_tr']['dmcol_idx']
        for tr, start, end in trialinds:
            tmpglm.dm[start:end, pl_idx][tmpglm.dm[start:end, pl_idx] > 0] = tmp_df.pLeft_last[tr]
            pl_old = fitinfo.probabilityLeft[tr]
            pll_old = fitinfo.pLeft_last[tr]
            tmpglm.dm[start:end, plt_idx][tmpglm.dm[start:end, plt_idx] == pl_old] = tmp_df.probabilityLeft[tr]
            tmpglm.dm[start:end, plt_idx][tmpglm.dm[start:end, plt_idx] == pll_old] = tmp_df.pLeft_last[tr]
        tmpglm.fit(method=method, printcond=False)
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
    currdate = str(date.today())
    # currdate = '2021-04-13'
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

    print(f'Fitting {len(sessions)} sessions...')

    for sessid in sessions:
        info = one.get_details(sessid)

        nickname = info['subject']
        sessdate = info['start_time'][:10]
        print(f'\nWorking on {nickname} from {sessdate}\n')
        for i in range(len(sessions[sessid])):
            probe = i
            filename = savepath + f'{nickname}/{sessdate}_session_{currdate}_probe{probe}_fit.p'
            if not os.path.exists(savepath + nickname):
                os.mkdir(savepath + nickname)
            subpaths = [n for n in glob(os.path.abspath(filename)) if os.path.isfile(n)]
            if len(subpaths) != 0:
                print(f'Skipped {nickname}, {sessdate}, probe{probe}: already fit.')
            if len(subpaths) == 0:
                print(sessid)
                if not offline:
                    _ = one.load(sessid, download_only=True)
                try:
                    outtup = fit_session(sessid, kernlen, nbases,
                                         prior_estimate=prior_estimate, stepwise=stepwise,
                                         probe_idx=probe, method=method,
                                         binwidth=binwidth, blocktrain=blocking,
                                         wholetrial_step=wholetrial_step,
                                         abswheel=abswheel, no_50perc=no_50perc)
                    nglm, realscores, scoreslist, weightslist = outtup
                except Exception as err:
                    tb = traceback.format_exc()
                    print(f'Subject {nickname} on {sessdate} failed:\n', type(err), err)
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
