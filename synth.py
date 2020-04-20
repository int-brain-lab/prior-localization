import numpy as np
from numpy.random import normal


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
    for start, end, prior, contrast in zip(stimtimes, fdbktimes, priors, contrasts):
        trial_len = int(np.ceil((end + 0.4 + 0.6) / SIMBINSIZE))
        stimarr = np.zeros(trial_len)
        fdbkarr = np.zeros(trial_len)
        stimarr[int(np.round(start / SIMBINSIZE))] = 1 * contrast
        fdbkarr[int(np.round(end / SIMBINSIZE))] = 1
        stimarr = np.convolve(stimkern * SIMBINSIZE, stimarr)[:trial_len]
        fdbkarr = np.convolve(fdbkkern * SIMBINSIZE, fdbkarr)[:trial_len]
        fdbkind = int(np.round(end / SIMBINSIZE))
        priorarr = np.array([prior] * fdbkind + [prior + .05] * (trial_len - fdbkind))
        priorarr = pgain * priorarr
        kernsum = priorarr + stimarr + fdbkarr
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
                spike_times.append(curr_t + noisevals[j] * SIMBINSIZE / 8)
        trialspikes.append(spike_times)
    return trialspikes, contrasts, priors, stimtimes, fdbktimes


if __name__ == "__main__":
    cell = 'cell17'
    gain = 10  # Hz
    perc_corr = 0.9
    tot_trials = 1000
    fitdata = np.load('./fits/ZM_2240/2020-01-22_session_2020-04-15_probe0_fit.p',
                      allow_pickle=True)
    fits = fitdata['fits']
    stimonL_kern = fits['stim_L'][cell]
    stimonR_kern = fits['stim_R'][cell]
    fdbkcorr_kern = fits['fdbck_corr'][cell]
    fdbkincorr_kern = fits['fdbck_incorr'][cell]
    priorgain = fits['prior'][cell]
    leftcorrtrials = simulate_cell(stimonL_kern, fdbkcorr_kern, priorgain, gain,
                                   int(tot_trials * 0.5 * perc_corr))
    leftincorrtrials = simulate_cell(stimonL_kern, fdbkincorr_kern, priorgain, gain,
                                     int(tot_trials * 0.5 * (1 - perc_corr)))
    rightcorrtrials = simulate_cell(stimonR_kern, fdbkcorr_kern, priorgain, gain,
                                    int(tot_trials * 0.5 * perc_corr))
    rightincorrtrials = simulate_cell(stimonR_kern, fdbkincorr_kern, priorgain, gain,
                                      int(tot_trials * 0.5 * (1 - perc_corr)))
                                     
