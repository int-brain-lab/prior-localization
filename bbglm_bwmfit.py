"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
from brainbox.modeling import glm
from export_funs import trialinfo_to_df
from prior_funcs import fit_sess_psytrack
from datetime import date
import pickle
import os

offline = True
one = one.ONE()


def fit_session(session_id, subject_name, sessdate, kernlen, nbases,
                t_before=0.4, t_after=0.6, prior_estimate='psytrack', max_len=2., probe_idx=0,
                method='minimize', alpha=0):
    trialsdf = trialinfo_to_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after)
    if prior_estimate == 'psytrack':
        print('Fitting psytrack esimates...')
        wts, stds = fit_sess_psytrack(session_id, maxlength=max_len, as_df=True)
    else:
        raise NotImplementedError('Only psytrack currently available')
    spk_times = one.load(session_id, dataset_types=['spikes.times'], offline=offline)[probe_idx]
    spk_clu = one.load(session_id, dataset_types=['spikes.clusters'], offline=offline)[probe_idx]
    fitinfo = pd.concat((trialsdf, wts['bias']), axis=1)
    bias_next = np.roll(fitinfo['bias'], -1)
    bias_next = pd.Series(bias_next, index=fitinfo['bias'].index)[:-1]
    fitinfo['bias_next'] = bias_next
    fitinfo = fitinfo.iloc[1:-1]
    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
                'feedbackType': 'value',
                'feedback_times': 'timing',
                'contrastLeft': 'value',
                'contrastRight': 'value',
                'goCue_times': 'timing',
                'stimOn_times': 'timing',
                'trial_start': 'timing',
                'trial_end': 'timing',
                'bias': 'value',
                'bias_next': 'value',
                'wheel_velocity': 'continuous'}
    nglm = glm.NeuralGLM(fitinfo, spk_times, spk_clu, vartypes)

    def stepfunc(row):
        currvec = np.ones(nglm.binf(row.stimOn_times)) * row.bias
        nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.stimOn_times)) * row.bias_next
        return np.hstack((currvec, nextvec))

    cosbases_long = glm.full_rcos(kernlen, nbases, nglm.binf)
    cosbases_short = glm.full_rcos(0.4, nbases, nglm.binf)
    nglm.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
                              cond=lambda tr: np.isfinite(tr.contrastLeft),
                              desc='Kernel conditioned on L stimulus onset')
    nglm.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
                              cond=lambda tr: np.isfinite(tr.contrastRight),
                              desc='Kernel conditioned on R stimulus onset')
    nglm.add_covariate_timing('correct', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == 1,
                              desc='Kernel conditioned on correct feedback')
    nglm.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == -1,
                              desc='Kernel conditioned on incorrect feedback')
    nglm.add_covariate_raw('prior', stepfunc, desc='Step function on prior estimate')
    nglm.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
    nglm.compile_design_matrix()
    nglm.bin_spike_trains()
    nglm.fit(method=method, alpha=1e-3)
    combined_weights = nglm.combine_weights()
    return nglm, combined_weights


if __name__ == "__main__":
    from ibl_pipeline import subject, ephys
    from ibl_pipeline.analyses import behavior as behavior_ana
    from glob import glob
    currdate = str(date.today())
    sessions = subject.Subject * subject.SubjectProject * ephys.acquisition.Session *\
        ephys.ProbeTrajectory * behavior_ana.SessionTrainingStatus
    bwm_sess = sessions & 'subject_project = "ibl_neuropixel_brainwide_01"' &\
        'good_enough_for_brainwide_map = 1'
    sessinfo = [info for info in bwm_sess]
    kernlen = 0.6
    nbases = 10
    method = 'sklearn'

    for s in sessinfo:
        sessid = str(s['session_uuid'])
        nickname = s['subject_nickname']
        sessdate = str(s['session_start_time'].date())
        probe = s['probe_idx']
        filename = f'./fits/{nickname}/{sessdate}_session_{currdate}_probe{probe}_fit.p'
        if not os.path.exists(f'./fits/{nickname}'):
            os.mkdir(f'./fits/{nickname}')
        subpaths = [n for n in glob(os.path.abspath(filename)) if os.path.isfile(n)]
        if len(subpaths) == 0:
            try:
                nglm, sessweights = fit_session(sessid, nickname, sessdate, kernlen, nbases,
                                                probe_idx=probe, method=method)
            except Exception as err:
                print(f'Subject {nickname} on {sessdate} failed:\n', type(err), err)
                continue

            outdict = {'kernlen': kernlen, 'nbases': nbases, 'weights': sessweights,
                       'fitobj': nglm}
            today = str(date.today())
            subjfilepath = os.path.abspath(filename)
            fw = open(subjfilepath, 'wb')
            pickle.dump(outdict, fw)
            fw.close()
