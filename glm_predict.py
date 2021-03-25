import numpy as np
import matplotlib.pyplot as plt
from brainbox.modeling.glm import NeuralGLM
from brainbox.modeling.glm_linear import LinearGLM
from brainbox.plot import peri_event_time_histogram


def predict(nglm, targ_regressors=None, trials=None, retlab=False, incl_bias=True):
    if trials is None:
        trials = nglm.trialsdf.index
    if targ_regressors is None:
        targ_regressors = nglm.covar.keys()
    dmcols = np.hstack([nglm.covar[r]['dmcol_idx'] for r in targ_regressors])
    dmcols = np.sort(dmcols)
    trlabels = nglm.trlabels
    trfilter = np.isin(trlabels, trials).flatten()
    w = nglm.coefs
    b = nglm.intercepts
    dm = nglm.dm[trfilter, :][:, dmcols]
    if type(nglm) == NeuralGLM:
        link = np.exp
    elif type(nglm) == LinearGLM:
        def link(x):
            return x
    else:
        raise TypeError('nglm must be an instance of NeuralGLM or LinearGLM')
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
        trials = nglm.trialsdf.index
    times = nglm.trialsdf[align_time].apply(nglm.binf)
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
    def __init__(self, nglm, trialsdf, trials, spk_t, spk_clu):
        self.covar = list(nglm.covar.keys())
        self.nglm = nglm
        self.binnedspikes = nglm.binnedspikes
        self.trialsdf = trialsdf
        self.spk_t = spk_t
        self.spk_clu = spk_clu
        self.trials = trials
        self.full_psths = {}
        self.cov_psths = {}
        self.combweights = nglm.combine_weights()

    def psth_summary(self, align_time, unit, t_before=0.1, t_after=0.6, ax=None):
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(8, 12))
        
        times = self.trialsdf.loc[self.trials, align_time]
        peri_event_time_histogram(self.spk_t, self.spk_clu,
                                  times,
                                  unit, t_before, t_after, bin_size=self.nglm.binwidth,
                                  error_bars='sem', ax=ax[0], smoothing=0.01)
        keytuple = (align_time, t_before, t_after)
        if keytuple not in self.full_psths:
            self.full_psths[keytuple] = pred_psth(self.nglm, align_time, t_before, t_after)
            self.cov_psths[keytuple] = {}
            tmp = self.cov_psths[keytuple]
            for cov in self.covar:
                tmp[cov] = pred_psth(self.nglm, align_time, t_before, t_after, [cov], self.trials,
                                     incl_bias=False)
                ax[2].plot(self.combweights[cov].loc[unit])
        x = np.arange(-t_before, t_after, self.nglm.binwidth)
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
