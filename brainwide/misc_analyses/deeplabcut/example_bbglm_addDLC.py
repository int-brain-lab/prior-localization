"""
Created on Mon June 28 2021
@author: Nate Miska
example script showing usage of DLC data for brainbox GLM fitting
"""
#need to cd to 'prior-localization' directory before running
from oneibl import one
import numpy as np
import pandas as pd
import neurencoding.utils as mut
from neurencoding import utils
from neurencoding import linear
from neurencoding import poisson
from neurencoding.design_matrix import DesignMatrix
from glm_predict import predict, pred_psth, GLMPredictor
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
import matplotlib.pyplot as plt

offline = False
one = one.ONE()

ephys_cache = {}

#load function first
def fit_session(session_id, kernlen, nbases, glm_type,
                t_before=1., t_after=0.6, prior_estimate=None, max_len=2., probe_idx=0,
                method='minimize', alpha=0, contnorm=5., wholetrial_step=False,
                abswheel=False, no_50perc=False, num_pseudosess=100,
                fit_intercept=True):
    BINSIZE = 0.02
    KERNLEN = 0.6
    SHORT_KL = 0.4
    NBASES = 10
    
    if not abswheel:
        signwheel = True
    else:
        signwheel = False
    trialsdf = bbone.load_trials_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after,
                                ret_abswheel=abswheel, ret_wheel=signwheel, ext_DLC=True)

    contrasts_left = list(trialsdf['contrastLeft'])
    contrasts_right = list(trialsdf['contrastRight'])
    indicestoremove = np.empty(len(contrasts_right),)
    indicestoremove[:] = np.NaN
    for j in np.arange(0,np.size(contrasts_right)):
        if contrasts_right[j] != 1 and contrasts_right[j] != 0.25:###conditional statement
            indicestoremove[j] = int(j)
            if j in trialsdf.index:
                trialsdf = trialsdf.drop(index = int(j))
    if prior_estimate == 'psytrack':
        print('Fitting psytrack esimates...')
        wts, stds = fit_sess_psytrack(session_id, maxlength=max_len, as_df=True)
        wts['bias'] = wts['bias'] - np.mean(wts['bias'])
        fitinfo = pd.concat((trialsdf, wts['bias']), axis=1)
        bias_next = np.roll(fitinfo['bias'], -1)
        bias_next = pd.Series(bias_next, index=fitinfo['bias'].index)[:-1]
        fitinfo['bias_next'] = bias_next
    elif prior_estimate is None:
        fitinfo = trialsdf.copy() #change name of trialsdf object
    else:
        raise NotImplementedError('Only psytrack currently available')
    # spk_times = one.load(session_id, dataset_types=['spikes.times'], offline=offline)[probe_idx]
    # spk_clu = one.load(session_id, dataset_types=['spikes.clusters'], offline=offline)[probe_idx]

    # A bit of messy loading to get spike times, clusters, and cluster brain regions.
    # This is the way it is because loading with regions takes forever. The weird for loop
    # ensures that we don't waste memory storing unnecessary and large arrays.
    probestr = 'probe0' + str(probe_idx)
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                aligned=True)
    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    clu_regions = clusters[probestr].acronym
    fitinfo['pLeft_last'] = pd.Series(np.roll(fitinfo['probabilityLeft'], 1),
                                        index=fitinfo.index)[:-1]
    fitinfo = fitinfo.iloc[1:-1]
    fitinfo['adj_contrastLeft'] = np.tanh(contnorm * fitinfo['contrastLeft']) / np.tanh(contnorm)
    fitinfo['adj_contrastRight'] = np.tanh(contnorm * fitinfo['contrastRight']) / np.tanh(contnorm)
    #adj_contrast is the pseudo-linearized contrasts

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
                'wheel_velocity': 'continuous',
                'DLC_Lpaw_xvel_leftcam': 'continuous',
                # 'DLC_Lpaw_y_leftcam': 'continuous', 
                'DLC_Rpaw_xvel_leftcam': 'continuous',}
                # 'DLC_Rpaw_y_leftcam': 'continuous',
                # 'DLC_Lpaw_x_rightcam': 'continuous',
                # 'DLC_Lpaw_y_rightcam': 'continuous',
                # 'DLC_Rpaw_x_rightcam': 'continuous',
                # 'DLC_Rpaw_y_rightcam': 'continuous'}
    if t_before < 0.7:
        raise ValueError('t_before needs to be 0.7 or greater in order to do -0.1 to -0.7 step'
                            ' function on pLeft')

    def stepfunc(row):
        currvec = np.ones(linglm.binf(row.duration)) * row.pLeft_last
        # nextvec = np.ones(linglm.binf(row.duration) - linglm.binf(row.feedback_times)) *\
        #     row.probabilityLeft
        return currvec

    def stepfunc_prestim(row):
        stepvec = np.zeros(linglm.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.pLeft_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.ones(linglm.binf(row.duration))
        currtr_start = linglm.binf(row.stimOn_times + 0.1)
        currtr_end = linglm.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.pLeft_last
        zerovec[currtr_end:] = row.probabilityLeft
        return zerovec

    def stepfunc_bias(row):
        currvec = np.ones(linglm.binf(row.feedback_times)) * row.bias
        nextvec = np.ones(linglm.binf(row.duration) - linglm.binf(row.feedback_times)) *\
            row.bias_next
        return np.hstack((currvec, nextvec))

    # Initialize design matrix
    design = DesignMatrix(fitinfo, vartypes=vartypes, binwidth=BINSIZE)
    # Build some basis functions
    longbases = mut.full_rcos(KERNLEN, NBASES, design.binf)
    shortbases = mut.full_rcos(SHORT_KL, NBASES, design.binf)

    # cosbases_long = utils.full_rcos(kernlen, nbases, linglm.binf) #cosbases defines number of cosine humps to use
    # cosbases_short = utils.full_rcos(0.4, 3, linglm.binf)


    design.add_covariate_timing('stimL', 'stimOn_times', longbases,
                            cond=lambda tr: np.isfinite(tr.contrastLeft),
                            desc='Stimulus onset left side kernel')
    design.add_covariate_timing('stimR', 'stimOn_times', longbases,
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                desc='Stimulus onset right side kernel')
    design.add_covariate_timing('correct', 'feedback_times', longbases,
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect', 'feedback_times', longbases,
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')

    # if prior_estimate is None and wholetrial_step:
    #     design.add_covariate_raw('pLeft', stepfunc, desc='Step function on prior estimate')
    # elif prior_estimate is None and not wholetrial_step:
    #     design.add_covariate_raw('pLeft', stepfunc_prestim,
    #                                 desc='Step function on prior estimate')
    #     # linglm.add_covariate_raw('pLeft_tr', stepfunc_poststim,
    #     #                          desc='Step function on post-stimulus prior')
    # elif prior_estimate == 'psytrack':
    #     design.add_covariate_raw('pLeft', stepfunc_bias, desc='Step function on prior estimate')

    # design.add_covariate('wheel', fitinfo['wheel_velocity'], shortbases, offset=-SHORT_KL,
    #                  desc='Anti-causal regressor for wheel velocity')
    design.add_covariate('DLC_L_xvel_leftcam', fitinfo['DLC_Lpaw_xvel_leftcam'], shortbases, offset=-SHORT_KL,
                     desc='Regressor for paw velocity calculated from DLC')
    # design.add_covariate('DLC_R_xvel_leftcam', fitinfo['DLC_Rpaw_xvel_leftcam'], shortbases, offset=-SHORT_KL,
    #                  desc='Regressor for paw velocity calculated from DLC')
    
    design.compile_design_matrix()

    if glm_type == 'linear':
        linglm = linear.LinearGLM(design, spk_times, spk_clu, binwidth=BINSIZE)
    else:
        linglm = poisson.PoissonGLM(design, spk_times, spk_clu, binwidth=BINSIZE)

    linglm.clu_regions = clu_regions
    stepbounds = [linglm.binf(0.1), linglm.binf(0.6)]

    if glm_type == 'poisson':
        linglm.fit()#method=method, alpha=0)
    else:
        linglm.fit()#method='pure', multi_score=True)

    return linglm

###example below for fitting a single session and visualizing GLM kernels for a specific unit
eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'#'03cf52f6-fba6-4743-a42e-dd1ac3072343'
probe_id = 'probe01'
if probe_id == 'probe00':
    probe_idx = 0
else:
    probe_idx = 1
fit_object = fit_session(session_id = eid, kernlen = 0.4, nbases = 10, glm_type = 'poisson', probe_idx=probe_idx)
#this object contains fit data, use fit_object. + tab to see all properties
spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid, one=one, aligned=True)
spk_times = spikes[probe_id].times #these should be output by fit_session to remove redundancy
spk_clu = spikes[probe_id].clusters
pred = GLMPredictor(fit_object, spk_times, spk_clu)
pred.psth_summary('stimOn_times', 12)
plt.show()

def filter_trials(stim_choice, stim_contrast, stim_side):
    if (stim_choice == 'all') & (stim_contrast == 'all') & (stim_side == 'both'):
        trials_id = np.arange(len(trials.choice))
    if (stim_choice == 'all') & (stim_contrast == 'all') & (stim_side != 'both'):
        contrast = 'contrastRight' if stim_side == 'right' else 'contrastLeft'
        trials_id = np.where(np.isfinite(trials[contrast]))[0]
    if (stim_choice == 'all') & (stim_contrast != 'all') & (stim_side == 'both'):
        trials_id = np.where((trials['contrastRight'] == stim_contrast) | (trials['contrastLeft'] == stim_contrast))[0]
    if (stim_choice != 'all') & (stim_contrast == 'all') & (stim_side == 'both'):
        outcome = 1 if stim_choice == 'correct' else -1
        trials_id = np.where(trials['feedbackType'] == outcome)[0]
    if (stim_choice == 'all') & (stim_contrast != 'all') & (stim_side != 'both'):
        contrast = 'contrastRight' if stim_side == 'right' else 'contrastLeft'
        trials_id = np.where(trials[contrast] == stim_contrast)[0]
    if (stim_choice != 'all') & (stim_contrast == 'all') & (stim_side != 'both'):
        contrast = 'contrastRight' if stim_side == 'right' else 'contrastLeft'
        outcome = 1 if stim_choice == 'correct' else -1
        trials_id = np.where((trials['feedbackType'] == outcome) & (np.isfinite(trials[contrast])))[0]  
    if  (stim_choice != 'all') & (stim_contrast != 'all') & (stim_side == 'both'):
        outcome = 1 if stim_choice == 'correct' else -1
        trials_id = np.where(((trials['contrastLeft'] == stim_contrast) | (trials['contrastRight'] == stim_contrast)) & (trials['feedbackType'] == outcome))[0]
    if (stim_choice != 'all') & (stim_contrast != 'all') & (stim_side != 'both'):
        contrast = 'contrastRight' if stim_side == 'right' else 'contrastLeft'
        outcome = 1 if stim_choice == 'correct' else -1
        trials_id = np.where((trials[contrast] == stim_contrast) & (trials['feedbackType'] == outcome))[0]
    return trials_id

def predict(nglm, targ_regressors=None, trials=None, retlab=False, incl_bias=True, glm_type='poisson'): #wasn't sure the best way to specify glm_type here...
    if trials is None:
        trials = nglm.design.trialsdf.index
    if targ_regressors is None:
        targ_regressors = nglm.design.covar.keys()
    dmcols = np.hstack([nglm.design.covar[r]['dmcol_idx'] for r in targ_regressors])
    dmcols = np.sort(dmcols)
    trlabels = nglm.design.trlabels
    trfilter = np.isin(trlabels, trials).flatten()
    w = nglm.coefs
    b = nglm.intercepts
    dm = nglm.design.dm[trfilter, :][:, dmcols]
    # if type(nglm) == NeuralGLM:
    #     link = np.exp
    # elif type(nglm) == LinearGLM:
    #     def link(x):
    #         return x
    if glm_type == 'poisson':
        link = np.exp
    elif glm_type == 'linear':
        def link(x):
            return x
    else:
        raise TypeError('nglm must be poisson or linear')
    if incl_bias:
        pred = {cell: link(dm @ w.loc[cell][dmcols] + b.loc[cell]) for cell in w.index}
    else:
        pred = {cell: link(dm @ w.loc[cell][dmcols]) for cell in w.index}
    # if type(nglm) == LinearGLM:
    #     for cell in pred:
    #         cellind = np.argwhere(nglm.clu_ids == cell)[0][0]
    #         pred[cell] += np.mean(nglm.binnedspikes[:, cellind])
    if not retlab:
        return pred
    else:
        return pred, trlabels[trfilter].flatten()

def pred_psth(nglm, align_time, t_before, t_after, targ_regressors=None, trials=None,
              incl_bias=True):
    if trials is None:
        trials = nglm.design.trialsdf.index
    times = nglm.design.trialsdf[align_time].apply(nglm.binf)
    tbef_bin = nglm.binf(t_before)
    taft_bin = nglm.binf(t_after)
    pred, labels = predict(nglm, targ_regressors, trials, retlab=True, incl_bias=incl_bias)
    t_inds = [np.searchsorted(labels, tr) + times[tr] for tr in trials]
    winds = [(t - tbef_bin, t + taft_bin) for t in t_inds]
    psths = {}
    for cell in pred.keys():
        cellpred = pred[cell]
        windarr = np.vstack([cellpred[w[0]:w[1]] for w in winds])
        psths[cell] = (np.mean(windarr, axis=0) / nglm.binwidth,
                       np.std(windarr, axis=0) / nglm.binwidth)
    return psths

class GLMPredictor:
    def __init__(self, nglm, spk_t, spk_clu):
        self.covar = list(nglm.design.covar.keys())
        self.nglm = nglm
        self.binnedspikes = nglm.binnedspikes
        self.design = nglm.design
        self.spk_t = spk_t
        self.spk_clu = spk_clu
        self.trials = nglm.design.trialsdf.index
        self.trialsdf = nglm.design.trialsdf #maybe not best way to do this
        self.full_psths = {}
        self.cov_psths = {}
        self.combweights = nglm.combine_weights()

    def psth_summary(self, align_time, unit, t_before=0.1, t_after=0.6, ax=None):
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(8, 12))

        times = self.trialsdf.loc[self.trials, align_time] #
        peri_event_time_histogram(self.spk_t, self.spk_clu,
                                  times,
                                  unit, t_before, t_after, bin_size=self.nglm.binwidth,
                                  error_bars='sem', ax=ax[0], smoothing=0.01)
        keytuple = (align_time, t_before, t_after)
        if keytuple not in self.full_psths:
            self.full_psths[keytuple] = pred_psth(self.nglm, align_time, t_before, t_after,
                                                  trials=self.trials)
            self.cov_psths[keytuple] = {}
            tmp = self.cov_psths[keytuple]
            for cov in self.covar:
                tmp[cov] = pred_psth(self.nglm, align_time, t_before, t_after,
                                     targ_regressors=[cov], trials=self.trials,
                                     incl_bias=False)
                ax[2].plot(self.combweights[cov].loc[unit])
        x = np.arange(-t_before, t_after, self.nglm.binwidth) + 0.01
        ax[0].plot(x, self.full_psths[keytuple][unit][0], label='Model prediction')
        ax[0].legend()
        for cov in self.covar:
            ax[1].plot(x, self.cov_psths[keytuple][cov][unit][0], label=cov)
        ax[1].set_title('Individual component contributions')
        ax[1].legend()
        if hasattr(self.nglm, 'clu_regions'):
            unitregion = self.nglm.clu_regions[unit]
            plt.suptitle(f'Unit {unit} from region {unitregion}')
        else:
            plt.suptitle(f'Unit {unit}')
        plt.tight_layout()
        return ax