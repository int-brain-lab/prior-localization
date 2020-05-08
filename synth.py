import numpy as np
from numpy.random import normal
from datetime import date
from scipy.io import savemat
from iofuns import loadmat
from fitplot_funs import err_wght_sync
import os


NUMCELLS = 30
SIMBINSIZE = 0.005
rt_vals = np.array([0.20748797, 0.39415191, 0.58081585, 0.76747979, 0.95414373,
                    1.14080767, 1.32747161, 1.51413555, 1.70079949, 1.88746343])
rt_probs = np.array([0.15970962, 0.50635209, 0.18693285, 0.0707804, 0.02540835,
                     0.01633394, 0.00907441, 0.00725953, 0.00544465, 0.01270417])
priorvals = np.linspace(-3, 3, 20)
priorprobs = np.ones(20) * (1 / 20)


def simulate_cell(stimkern, fdbkkern, pgain, gain, num_trials=500):
    stimtimes = np.ones(num_trials) * 0.4
    fdbktimes = np.random.choice(rt_vals, size=num_trials, p=rt_probs) \
        + stimtimes + normal(size=num_trials) * 0.05
    priors = np.random.choice(priorvals, size=num_trials, p=priorprobs)
    contrasts = np.random.uniform(size=num_trials)
    trialspikes = []
    trialrange = range(num_trials)
    for i, start, end, prior, contrast in zip(trialrange, stimtimes, fdbktimes, priors, contrasts):
        trial_len = int(np.ceil((end + 0.6) / SIMBINSIZE))
        stimarr = np.zeros(trial_len)
        fdbkarr = np.zeros(trial_len)
        stimarr[int(np.round(start / SIMBINSIZE))] = 1 * contrast
        fdbkarr[int(np.round(end / SIMBINSIZE))] = 1
        stimarr = np.convolve(stimkern, stimarr)[:trial_len - 1]
        fdbkarr = np.convolve(fdbkkern, fdbkarr)[:trial_len - 1]
        fdbkind = int(np.round(end / SIMBINSIZE))
        try:
            priorarr = np.array([prior] * fdbkind + [priors[i + 1]] * (trial_len - fdbkind))
        except IndexError:
            continue
        priorarr = pgain * priorarr
        kernsum = priorarr[:-1] + stimarr + fdbkarr
        ratevals = (np.exp(kernsum) + gain) * SIMBINSIZE
        spikecounts = np.random.poisson(ratevals)
        spike_times = []
        noisevals = normal(size=np.max(spikecounts))
        for i in np.nonzero(spikecounts)[0]:
            if i == 0:
                curr_t = SIMBINSIZE / 4
            else:
                curr_t = i * SIMBINSIZE
            for j in range(spikecounts[i]):
                jitterspike = curr_t + noisevals[j] * SIMBINSIZE / 8
                if jitterspike < 0:
                    jitterspike = 0
                spike_times.append(jitterspike)
        trialspikes.append(spike_times)
    return trialspikes, contrasts, priors, stimtimes, fdbktimes


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # import brainbox.plot as bbp
    from scipy.interpolate import interp1d
    cell = 'cell601'
    gain = 1  # Hz
    perc_corr = 0.6
    tot_trials = 10000
    fitdata = np.load('./fits/ZM_2240/2020-01-21_session_2020-04-24_probe0_fit.p',
                      allow_pickle=True)
    fitbinsize = 0.02
    kernel_t = np.arange(0, 0.6, fitbinsize)
    sim_t = np.arange(0, 0.6 - fitbinsize + 1e-10, SIMBINSIZE)
    fits = fitdata['fits']
    stimonL_kern = interp1d(kernel_t, fits['stim_L'][cell])(sim_t)
    stimonR_kern = interp1d(kernel_t, fits['stim_R'][cell])(sim_t)
    fdbkcorr_kern = interp1d(kernel_t, fits['fdbck_corr'][cell])(sim_t)
    fdbkincorr_kern = interp1d(kernel_t, fits['fdbck_incorr'][cell])(sim_t)
    priorgain = fits['prior'][cell]
    leftcorrtrials = simulate_cell(stimonL_kern, fdbkcorr_kern, priorgain, gain,
                                   int(tot_trials * 0.5 * perc_corr))
    leftincorrtrials = simulate_cell(stimonL_kern, fdbkincorr_kern, priorgain, gain,
                                     int(tot_trials * 0.5 * (1 - perc_corr)))
    rightcorrtrials = simulate_cell(stimonR_kern, fdbkcorr_kern, priorgain, gain,
                                    int(tot_trials * 0.5 * perc_corr))
    rightincorrtrials = simulate_cell(stimonR_kern, fdbkincorr_kern, priorgain, gain,
                                      int(tot_trials * 0.5 * (1 - perc_corr)))
    conds = [('Left', 1), ('Left', -1), ('Right', 1), ('Right', -1)]
    trialdata = [leftcorrtrials, leftincorrtrials, rightcorrtrials, rightincorrtrials]
    trials = []
    for cond, tr in zip(conds, trialdata):
        spikes, contr, priors, stimt, fdbkt = tr
        for i in range(len(spikes)):
            trialdict = {'spikes': np.array(spikes[i]),
                         'clu': np.ones(len(spikes[i])),
                         'contrastLeft': contr[i] if cond[0] == 'Left' else np.nan,
                         'contrastRight': contr[i] if cond[0] == 'Right' else np.nan,
                         'stimOn_times': stimt[i],
                         'feedback_times': fdbkt[i],
                         'prior': priors[i],
                         'feedbackType': cond[1]}
            trials.append(trialdict)
    currdate = str(date.today())
    outdict = {'subject_name': 'SYNTHETIC', 'trials': trials, 'clusters': np.array([1])}
    datafile = os.path.abspath(f'./data/SYNTHETIC_{currdate}_tmp.mat')
    savemat(datafile, outdict)
    os.system('matlab -nodisplay -r "iblglm_add2path; '
              f'fitsess(\'{datafile}\', 10, 0.02, 0.6); exit;"')
    fitdata = loadmat(f'./fits/SYNTHETIC_{currdate}_tmp_fit.mat')
    kernt = np.arange(0, 0.6, 0.02)
    simt = np.arange(0, 0.6, SIMBINSIZE)
    fig, axes = plt.subplots(5, 1)
    stLwts = fitdata['cellweights']['cell1']['stonL']['data']
    stLerr = np.sqrt(err_wght_sync(fitdata['cellstats']['cell1'][:10], stLwts))
    axes[0].fill_between(kernt, stLwts - stLerr, stLwts + stLerr, alpha=0.5, color='blue',
                         label='recovered L error')
    axes[0].plot(kernt, stLwts, label='recovered L')
    axes[0].plot(sim_t, stimonL_kern, label='True L')
    axes[0].legend()

    stRwts = fitdata['cellweights']['cell1']['stonR']['data']
    stRerr = np.sqrt(err_wght_sync(fitdata['cellstats']['cell1'][10:20], stRwts))
    axes[1].fill_between(kernt, stRwts - stRerr, stRwts + stRerr, alpha=0.5, color='blue',
                         label='recovered R error')
    axes[1].plot(kernt, stRwts, label='recovered R')
    axes[1].plot(sim_t, stimonR_kern, label='True R')
    axes[1].legend()

    fdCwts = fitdata['cellweights']['cell1']['fdbckCorr']['data']
    fdCerr = np.sqrt(err_wght_sync(fitdata['cellstats']['cell1'][20:30], fdCwts))
    axes[2].plot(sim_t, fdbkcorr_kern, label='True correct')
    axes[2].fill_between(kernt, fdCwts - fdCerr, fdCwts + fdCerr, alpha=0.5, color='orange',
                         label='recovered correct error')
    axes[2].plot(kernt, fdCwts, label='recovered correct')
    axes[2].legend()

    fdIwts = fitdata['cellweights']['cell1']['fdbckInc']['data']
    fdIerr = np.sqrt(err_wght_sync(fitdata['cellstats']['cell1'][30:40], fdIwts))
    axes[3].plot(sim_t, fdbkincorr_kern, label='True incorrect')
    axes[3].fill_between(kernt, fdIwts - fdIerr, fdIwts + fdIerr, alpha=0.5, color='orange',
                         label='recovered incorr error')
    axes[3].plot(kernt, fdIwts, label='recovered incorr')
    axes[3].legend()

    axes[4].bar([0, 1], [priorgain, fitdata['cellweights']['cell1']['prvec']['data']])
    plt.savefig('/home/berk/Documents/parameter_recovery.png', dpi=800)
