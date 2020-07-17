import numpy as np
from numpy.random import normal
from scipy.interpolate import interp1d
from brainbox.modeling.glm import convbasis

NUMCELLS = 30
SIMBINSIZE = 0.005
rt_vals = np.array([0.20748797, 0.39415191, 0.58081585, 0.76747979, 0.95414373,
                    1.14080767, 1.32747161, 1.51413555, 1.70079949, 1.88746343])
rt_probs = np.array([0.15970962, 0.50635209, 0.18693285, 0.0707804, 0.02540835,
                     0.01633394, 0.00907441, 0.00725953, 0.00544465, 0.01270417])
priorvals = np.linspace(-3, 3, 20)
priorprobs = np.ones(20) * (1 / 20)
wheelmotifs = np.load('wheelmotifs.p', allow_pickle=True)
t_b = 0.4
t_a = 0.6


def simulate_cell(stimkern, fdbkkern, pgain, gain, num_trials=500, exp=False):
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
        stimarr[int(np.ceil(start / SIMBINSIZE))] = 1
        fdbkarr[int(np.ceil(end / SIMBINSIZE))] = 1
        stimarr = np.convolve(stimkern, stimarr)[:trial_len]
        fdbkarr = np.convolve(fdbkkern, fdbkarr)[:trial_len]
        fdbkind = int(np.ceil(end / SIMBINSIZE))
        try:
            priorarr = np.array([prior] * fdbkind + [priors[i + 1]] * (trial_len - fdbkind))
        except IndexError:
            continue
        priorarr = pgain * priorarr
        kernsum = priorarr + stimarr + fdbkarr
        if exp:
            ratevals = (np.exp(kernsum) + gain) * SIMBINSIZE
        else:
            ratevals = (kernsum + gain) * SIMBINSIZE
        spikecounts = np.random.poisson(ratevals)
        spike_times = []
        noisevals = normal(loc=SIMBINSIZE / 2, scale=SIMBINSIZE / 8, size=np.max(spikecounts))
        for i in np.nonzero(spikecounts)[0]:
            if i == 0:
                curr_t = SIMBINSIZE / 4
            else:
                curr_t = i * SIMBINSIZE
            for j in range(spikecounts[i]):
                jitterspike = curr_t + noisevals[j]
                if jitterspike < 0:
                    jitterspike = 0
                spike_times.append(jitterspike)
        trialspikes.append(spike_times)
    return trialspikes, contrasts, priors, stimtimes, fdbktimes


def simulate_cell_realdf(trialsdf, stimkern, fdbkkern, wheelkern, pgain, gain, numtrials=None):
    trialspikes = []
    if 'bias' not in trialsdf.columns:
        raise KeyError('Trialsdf must have bias columns of prior estimates')
    if numtrials is not None:
        if numtrials > len(trialsdf.index):
            raise ValueError('numtrials must be less than number of trialsdf rows')
        keeptrials = np.random.choice(range(len(trialsdf) - 1), numtrials, replace=False)
    else:
        keeptrials = trialsdf.index[:-1]
    for indx in keeptrials:
        tr = trialsdf.iloc[indx]
        start = t_b
        end = tr.feedback_times - tr.trial_start
        prior = tr.bias
        newpr = trialsdf.iloc[indx + 1].bias
        wheel = trialsdf.iloc[indx].wheel_velocity
        trial_len = np.ceil((end + t_a) / SIMBINSIZE).astype(int)
        wheelinterp = interp1d(np.arange(len(wheel)) * 0.02, wheel, fill_value='extrapolate')
        wheelnew = wheelinterp(np.arange(trial_len) * SIMBINSIZE)

        stimarr = np.zeros(trial_len)
        fdbkarr = np.zeros(trial_len)
        stimarr[int(np.ceil(start / SIMBINSIZE))] = 1
        fdbkarr[int(np.ceil(end / SIMBINSIZE))] = 1
        stimarr = np.convolve(stimkern, stimarr)[:trial_len]
        fdbkarr = np.convolve(fdbkkern, fdbkarr)[:trial_len]
        wheelarr = convbasis(wheelnew.reshape(-1, 1),
                             wheelkern.reshape(-1, 1),
                             offset=-np.ceil(0.4 / SIMBINSIZE).astype(int))
        wheelarr = np.exp(wheelarr).flatten()
        fdbkind = int(np.ceil(end / SIMBINSIZE))
        priorarr = np.array([prior] * fdbkind + [newpr] * (trial_len - fdbkind))
        priorarr = pgain * np.exp(priorarr)
        kernsum = priorarr + stimarr + fdbkarr + wheelarr
        ratevals = (kernsum + gain) * SIMBINSIZE
        spikecounts = np.random.poisson(ratevals)
        spike_times = []
        noisevals = normal(loc=SIMBINSIZE / 2, scale=SIMBINSIZE / 8, size=np.max(spikecounts))
        for i in np.nonzero(spikecounts)[0]:
            if i == 0:
                curr_t = SIMBINSIZE / 4
            else:
                curr_t = i * SIMBINSIZE
            for j in range(spikecounts[i]):
                jitterspike = curr_t + noisevals[j]
                if jitterspike < 0:
                    jitterspike = 0
                spike_times.append(jitterspike)
        trialspikes.append(spike_times)
    return trialspikes, keeptrials


def stepfunc(row):
    currvec = np.ones(nglm.binf(row.stimOn_times)) * row.bias
    nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.stimOn_times)) * row.bias_next
    return np.hstack((currvec, nextvec))


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from brainbox.singlecell import calculate_peths
    from oneibl import one
    from export_funs import trialinfo_to_df
    from prior_funcs import fit_sess_psytrack
    from brainbox.modeling import glm

    one = one.ONE()
    cell_id = 0
    subject = 'ZM_2240'
    sessdate = '2020-01-22'
    ids = one.search(subject=subject, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.times'])
    trialsdf = trialinfo_to_df(ids[0], maxlen=2.)
    wts, stds = fit_sess_psytrack(ids[0], maxlength=2., as_df=True)
    trialsdf = pd.concat((trialsdf, wts['bias']), axis=1)
    spk_times = one.load(ids[0], dataset_types=['spikes.times'], offline=True)[0]
    spk_clu = one.load(ids[0], dataset_types=['spikes.clusters'], offline=True)[0]
    left_t = trialsdf[np.isfinite(trialsdf.contrastLeft)].stimOn_times
    fdbk = trialsdf[np.isfinite(trialsdf.contrastLeft)].feedback_times
    left_rate, _ = calculate_peths(spk_times, spk_clu, [cell_id], left_t, pre_time=0,
                                   post_time=0.6, bin_size=SIMBINSIZE)
    fdbk_rate, _ = calculate_peths(spk_times, spk_clu, [cell_id], fdbk, pre_time=0,
                                   post_time=0.6, bin_size=SIMBINSIZE)
    wheelkern = np.exp(-0.5 * ((np.linspace(0, 0.4, int(0.4 / SIMBINSIZE)) - 0.3) / 0.05)**2)
    wheelkern = wheelkern / np.max(wheelkern) * 4
    # fdbk_rate = Bunch(means=np.zeros_like(left_rate.means))
    fitkerns = []
    nvals = np.linspace(10, len(trialsdf) - 5, 6, dtype=int)
    trialsdf = trialsdf[np.isfinite(trialsdf.bias)]
    bias_next = np.roll(trialsdf['bias'], -1)
    bias_next = pd.Series(bias_next, index=trialsdf['bias'].index)[:-1]
    trialsdf['bias_next'] = bias_next
    pgainval = 2
    for ntrials in nvals:
        for i in range(20):
            trialspikes, indices = simulate_cell_realdf(trialsdf,
                                                        left_rate.means.flatten(),
                                                        fdbk_rate.means.flatten(),
                                                        wheelkern,
                                                        pgain=pgainval, gain=0,
                                                        numtrials=ntrials)
            adj_spkt = np.hstack([trialsdf.iloc[i].trial_start + np.array(t)
                                  for i, t in zip(indices, trialspikes)])
            sess_trialspikes = np.sort(adj_spkt)
            sess_clu = np.ones_like(adj_spkt, dtype=int)
            fitdf = trialsdf.iloc[indices][['trial_start', 'trial_end',
                                            'stimOn_times', 'feedback_times',
                                            'wheel_velocity','bias', 'bias_next']].sort_index()

            nglm = glm.NeuralGLM(fitdf, sess_trialspikes, sess_clu,
                                 {'trial_start': 'timing',
                                  'stimOn_times': 'timing',
                                  'feedback_times': 'timing',
                                  'trial_end': 'timing',
                                  'wheel_velocity': 'continuous',
                                  'bias': 'value',
                                  'bias_next': 'value'},
                                 mintrials=1, train=1.)
            bases = glm.raised_cosine(0.4, 10, nglm.binf)
            longbases = glm.raised_cosine(0.6, 10, nglm.binf)
            nglm.add_covariate_timing('fdbk', 'feedback_times', longbases, desc='synth fdbk')
            nglm.add_covariate_timing('stim', 'stimOn_times', longbases, desc='synth stimon')
            nglm.add_covariate('wheel', fitdf['wheel_velocity'], bases, offset=-0.4,
                               desc='synthetic wheel move')
            nglm.add_covariate_raw('prior', stepfunc, desc='Step function on prior estimate')
            nglm.compile_design_matrix()
            if np.linalg.cond(nglm.dm) > 1e6:
                print('Bad COND!')
                continue
            nglm.bin_spike_trains()
            nglm.fit(method='sklearn')
            combined_weights = nglm.combine_weights()
            fitkerns.append((ntrials,
                             combined_weights['wheel'].loc[1],
                             combined_weights['stim'].loc[1],
                             combined_weights['fdbk'].loc[1],
                             combined_weights['prior'].loc[1]))
    fig, ax = plt.subplots(7, 4)
    ax[0, 0].plot(np.linspace(0, 0.4, int(0.4 / SIMBINSIZE)),
                  wheelkern, label='Ground truth')
    ax[0, 1].plot(left_rate.tscale,
                  left_rate.means.flatten(), label='Ground truth')
    ax[0, 2].plot(fdbk_rate.tscale,
                  fdbk_rate.means.flatten(), label='Ground truth')
    ax[0, 3].remove()
    ax[0, 0].set_ylabel('Firing rate (Hz)')
    ax[0, 0].legend()
    ax[0, 0].set_title('Wheel kernel for generated spikes')
    ax[0, 1].set_title('Original PSTH taken from unit 0')
    ax[0, 2].set_title('Original PSTH taken from unit 0')
    colors = sns.light_palette('navy', 6)
    for i in range(0, 6):
        nkerns_wh = [x[1] for x in fitkerns if x[0] == nvals[i]]
        nkerns_stim = [x[2] for x in fitkerns if x[0] == nvals[i]]
        nkerns_fdbk = [x[3] for x in fitkerns if x[0] == nvals[i]]
        nkerns_pgain = [np.exp(x[4]) for x in fitkerns if x[0] == nvals[i]]
        nkerns_pgain.append(np.log(pgainval))
        for fit in nkerns_wh:
            ax[i + 1, 0].plot(fit, c=colors[i],)
        for fit in nkerns_stim:
            ax[i + 1, 1].plot(np.exp(fit), c=colors[i],)
        for fit in nkerns_fdbk:
            ax[i + 1, 2].plot(np.exp(fit), c=colors[i],)
        ax[i + 1, 3].bar(np.arange(len(nkerns_pgain)), nkerns_pgain,
                         color=['blue'] * 20 + ['orange'])
        ax[i + 1, 0].set_title(f'{nvals[i]} trials')
        ax[i + 1, 0].set_xlabel('Time (s)')
        ax[i + 1, 0].set_ylabel('Kernel weight')
        ax[i + 1, 1].set_title(f'{nvals[i]} trials')
        ax[i + 1, 1].set_xlabel('Time (s)')
        ax[i + 1, 1].set_ylabel('Kernel weight')
        ax[i + 1, 2].set_title(f'{nvals[i]} trials')
        ax[i + 1, 2].set_xlabel('Time (s)')
        ax[i + 1, 2].set_ylabel('Kernel weight')