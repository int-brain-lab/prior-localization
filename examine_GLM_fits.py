from iofuns import loadmat
import numpy as np
import matplotlib.pyplot as plt
from brainbox.core import Bunch
import brainbox.plot as bbp
from oneibl import one
import os
from datetime import date


def expnd_err(errors, weights):
    mult = int(weights.shape[0] / errors.shape[0])
    expnd = []
    _ = [expnd.extend([x] * mult) for x in errors]
    if len(expnd) != weights.shape[0]:
        diff = weights.shape[0] - len(expnd) 
        expnd.extend([errors[-1]] * diff)
    return np.array(expnd)


one = one.ONE()
animal = 'ZM_2240'
sessdate = '2020-01-24'
currfits = loadmat(f'./{animal}_{sessdate}_fit.mat')
cellweights = Bunch(currfits['cellweights'])
cellstats = Bunch(currfits['cellstats'])

sess_id = one.search(subject=animal, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.clusters'])
spikes, clus = one.load(sess_id[0], dataset_types=['spikes.times', 'spikes.clusters'])
stimon, rt, fdbck_t = one.load(sess_id[0], ['trials.stimOn_times',
                                            'trials.response_times',
                                            'trials.feedback_times'])
left, right = one.load(sess_id[0], ['trials.contrastLeft', 'trials.contrastRight'])
contrasts = np.vstack((left, right)).T
fittrials = np.isfinite(contrasts[:, 1]) & (contrasts[:, 1] > 0)

fitfolder = os.path.abspath(f'./data/{animal}_{sessdate}_fits/')
if not os.path.exists(fitfolder):
    os.mkdir(fitfolder)

fitcells = cellweights.keys()
for cell in fitcells:
    cellind = int(cell[4:])
    currwts = cellweights[cell]
    fig, axes = plt.subplots(4, 1, figsize=(6, 8))
    bbp.peri_event_time_histogram(spikes, clus, stimon[fittrials], cellind, t_after=0.6,
                                  t_before=0.01, ax=axes[0], error_bars='sem')
    axes[0].set_title('PSTH on Stim On')
    stimerr = expnd_err(cellstats[cell][:20], currwts['stimOn']['data'])
    axes[1].errorbar(currwts['stimOn']['tr'], currwts['stimOn']['data'], stimerr)
    axes[1].set_title('GLM weights for post-stimulus-on')
    # bbp.peri_event_time_histogram(spikes, clus, rt[fittrials], cellind, t_after=0.6, t_before=0.01,
    #                               ax=axes[2])
    # axes[2].set_title('PETH for response of animal')
    # rterr = expnd_err(cellstats[cell][10:20], currwts['resp_time']['data'])
    # axes[3].errorbar(currwts['resp_time']['tr'], currwts['resp_time']['data'], rterr)
    # axes[3].set_title('GLM weights for response time')
    bbp.peri_event_time_histogram(spikes, clus, fdbck_t[fittrials], cellind, t_after=0.6,
                                  t_before=0.01, ax=axes[2])
    axes[2].set_title('PSTH for feedback (reward or noise)')
    fdbckerr = expnd_err(cellstats[cell][10:-1], currwts['feedback_t']['data'])
    axes[3].errorbar(currwts['feedback_t']['tr'], currwts['feedback_t']['data'],
                     fdbckerr)
    axes[3].set_title('GLM weights for feedback')
    plt.tight_layout()
    plt.savefig(fitfolder + f'/{cell}_{date.today()}_fit.png', DPI=500)
    plt.close()
