import numpy as np
from numpy.random import normal
from scipy.interpolate import interp1d
from brainbox.modeling.glm import convbasis

NUMCELLS = 30
BINSIZE = 0.005
rt_vals = np.array([0.20748797, 0.39415191, 0.58081585, 0.76747979, 0.95414373,
                    1.14080767, 1.32747161, 1.51413555, 1.70079949, 1.88746343])
rt_probs = np.array([0.15970962, 0.50635209, 0.18693285, 0.0707804, 0.02540835,
                     0.01633394, 0.00907441, 0.00725953, 0.00544465, 0.01270417])
priorvals = np.linspace(-3, 3, 20)
priorprobs = np.ones(20) * (1 / 20)
# wheelmotifs = np.load('wheelmotifs.p', allow_pickle=True)
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
        trial_len = int(np.ceil((end + 0.6) / BINSIZE))
        stimarr = np.zeros(trial_len)
        fdbkarr = np.zeros(trial_len)
        stimarr[int(np.ceil(start / BINSIZE))] = 1
        fdbkarr[int(np.ceil(end / BINSIZE))] = 1
        stimarr = np.convolve(stimkern, stimarr)[:trial_len]
        fdbkarr = np.convolve(fdbkkern, fdbkarr)[:trial_len]
        fdbkind = int(np.ceil(end / BINSIZE))
        try:
            priorarr = np.array([prior] * fdbkind + [priors[i + 1]] * (trial_len - fdbkind))
        except IndexError:
            continue
        priorarr = pgain * priorarr
        kernsum = priorarr + stimarr + fdbkarr
        if exp:
            ratevals = (np.exp(kernsum) + gain) * BINSIZE
        else:
            ratevals = (kernsum + gain) * BINSIZE
        spikecounts = np.random.poisson(ratevals)
        spike_times = []
        noisevals = normal(loc=BINSIZE / 2, scale=BINSIZE / 8, size=np.max(spikecounts))
        for i in np.nonzero(spikecounts)[0]:
            if i == 0:
                curr_t = BINSIZE / 4
            else:
                curr_t = i * BINSIZE
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
        trial_len = np.ceil((end + t_a) / BINSIZE).astype(int)
        wheelinterp = interp1d(np.arange(len(wheel)) * 0.02, wheel, fill_value='extrapolate')
        wheelnew = wheelinterp(np.arange(trial_len) * BINSIZE)

        stimarr = np.zeros(trial_len)
        fdbkarr = np.zeros(trial_len)
        stimarr[int(np.ceil(start / BINSIZE))] = 1
        fdbkarr[int(np.ceil(end / BINSIZE))] = 1
        stimarr = np.convolve(stimkern, stimarr)[:trial_len]
        fdbkarr = np.convolve(fdbkkern, fdbkarr)[:trial_len]
        if not np.all(wheelkern == 0):
            wheelarr = convbasis(wheelnew.reshape(-1, 1),
                                 wheelkern.reshape(-1, 1),
                                 offset=-np.ceil(0.4 / BINSIZE).astype(int))
            wheelarr = np.exp(wheelarr).flatten()
        else:
            wheelarr = np.ones_like(stimarr) * 0.005
        fdbkind = int(np.ceil(end / BINSIZE))
        priorarr = np.array([prior] * fdbkind + [newpr] * (trial_len - fdbkind))
        priorarr = pgain * np.exp(priorarr)
        kernsum = priorarr + stimarr + fdbkarr + wheelarr
        ratevals = (kernsum + gain) * BINSIZE
        spikecounts = np.random.poisson(ratevals)
        spike_times = []
        noisevals = normal(loc=BINSIZE / 2, scale=BINSIZE / 8, size=np.max(spikecounts))
        for i in np.nonzero(spikecounts)[0]:
            if i == 0:
                curr_t = BINSIZE / 4
            else:
                curr_t = i * BINSIZE
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


def fit_sim(trialsdf, left_rate, fdbk_rate, wheelkern, ntrials, priorgain=0):
    fitbool = [test is not None for test in (left_rate, fdbk_rate, wheelkern)]
    if all(np.invert(fitbool)):
        raise TypeError('All kernels cannot be None')
    if left_rate is None:
        left_rate = np.zeros(int(0.6 / BINSIZE))
    if fdbk_rate is None:
        fdbk_rate = np.zeros(int(0.6 / BINSIZE))
    if wheelkern is None:
        wheelkern = np.zeros(int(0.4 / BINSIZE))
    trialspikes, indices = simulate_cell_realdf(trialsdf,
                                                left_rate,
                                                fdbk_rate,
                                                wheelkern,
                                                pgain=priorgain, gain=0,
                                                numtrials=ntrials)
    adj_spkt = np.hstack([trialsdf.iloc[i].trial_start + np.array(t)
                          for i, t in zip(indices, trialspikes)])
    sess_trialspikes = np.sort(adj_spkt)
    sess_clu = np.ones_like(adj_spkt, dtype=int)
    fitdf = trialsdf.iloc[indices][['trial_start', 'trial_end',
                                    'stimOn_times', 'feedback_times',
                                    'wheel_velocity', 'bias', 'bias_next']].sort_index()

    nglm = glm.NeuralGLM(fitdf, sess_trialspikes, sess_clu,
                         {'trial_start': 'timing',
                          'stimOn_times': 'timing',
                          'feedback_times': 'timing',
                          'trial_end': 'timing',
                          'wheel_velocity': 'continuous',
                          'bias': 'value',
                          'bias_next': 'value'},
                         mintrials=1, train=1.)
    bases = glm.full_rcos(0.4, 10, nglm.binf)
    longbases = glm.full_rcos(0.6, 10, nglm.binf)
    if fitbool[0] is True:
        nglm.add_covariate_timing('stim', 'stimOn_times', longbases, desc='synth stimon')
    if fitbool[1] is True:
        nglm.add_covariate_timing('fdbk', 'feedback_times', longbases, desc='synth fdbk')
    if fitbool[2] is True:
        nglm.add_covariate('wheel', fitdf['wheel_velocity'], bases, offset=-0.4,
                           desc='synthetic wheel move')
    if priorgain != 0:
        nglm.add_covariate_raw('prior', stepfunc, desc='Step function on prior estimate')

    nglm.compile_design_matrix()
    if np.linalg.cond(nglm.dm) > 1e6:
        print('Bad COND!')
        return None
    nglm.bin_spike_trains()
    nglm.fit(method='minimize', alpha=0)
    combined_weights = nglm.combine_weights()
    retlist = []
    retlist.append(nglm.intercepts.iloc[0])
    if fitbool[0] is True:
        retlist.append(combined_weights['stim'].loc[1])
    if fitbool[1] is True:
        retlist.append(combined_weights['fdbk'].loc[1])
    if fitbool[2] is True:
        retlist.append(combined_weights['wheel'].loc[1])
    return retlist


if __name__ == "__main__":
    import pickle
    import pandas as pd
    import itertools as it
    from tqdm import tqdm
    from brainbox.singlecell import calculate_peths
    from oneibl import one
    from export_funs import trialinfo_to_df
    from prior_funcs import fit_sess_psytrack
    from brainbox.modeling import glm

    one = one.ONE()
    subject = 'ZM_2240'
    sessdate = '2020-01-22'
    ids = one.search(subject=subject, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.times'])
    trialsdf = trialinfo_to_df(ids[0], maxlen=2.)
    wts, stds = fit_sess_psytrack(ids[0], maxlength=2., as_df=True)
    trialsdf = pd.concat((trialsdf, wts['bias']), axis=1)
    spk_times = one.load(ids[0], dataset_types=['spikes.times'], offline=True)[0]
    spk_clu = one.load(ids[0], dataset_types=['spikes.clusters'], offline=True)[0]

    trialsdf = trialsdf[np.isfinite(trialsdf.bias)]
    bias_next = np.roll(trialsdf['bias'], -1)
    bias_next = pd.Series(bias_next, index=trialsdf['bias'].index)[:-1]
    trialsdf['bias_next'] = bias_next

    cell_ids = [0, 1, 2, 9, 15,
                16, 20, 26, 29, 30,
                # 32, 34, 52, 71,
                ]
    kernelcombs = list(it.product(*[(False, True)] * 3))  # Boolean combination of kernels
    nvals = np.linspace(100, len(trialsdf) - 5, 3, dtype=int)
    _ = kernelcombs.pop(0)
    fits = {}
    for cell in tqdm(cell_ids, desc='Cell'):
        cellfits = {}
        for comb in tqdm(kernelcombs, desc='Kernel combination', leave=False):
            combfits = {}
            left_t = trialsdf[np.isfinite(trialsdf.contrastLeft)].stimOn_times
            fdbk = trialsdf[np.isfinite(trialsdf.contrastLeft)].feedback_times
            left_rate, _ = calculate_peths(spk_times, spk_clu, [cell], left_t, pre_time=0,
                                           post_time=0.6, bin_size=BINSIZE)
            left_rate.means = left_rate.means - left_rate.means.min()
            left_rate = left_rate.means.flatten()
            fdbk_rate, _ = calculate_peths(spk_times, spk_clu, [cell], fdbk, pre_time=0,
                                           post_time=0.6, bin_size=BINSIZE)
            fdbk_rate.means = fdbk_rate.means - fdbk_rate.means.min()
            fdbk_rate = fdbk_rate.means.flatten()
            wheelkern = np.exp(-0.5 * ((np.linspace(0, 0.4, int(0.4 / BINSIZE)) - 0.3) / 0.05)**2)
            logpeak = np.log(np.max([np.max(left_rate), np.max(fdbk_rate)]))
            wheelkern = wheelkern / np.max(wheelkern) * logpeak
            for N in tqdm(nvals, desc='N values', leave=False):
                combfits[N] = []
                for i in tqdm(range(5), desc='iter', leave=False):
                    stimk = left_rate if comb[0] else None
                    fdbkk = fdbk_rate if comb[1] else None
                    wheelk = wheelkern if comb[2] else None
                    currfit = fit_sim(trialsdf, stimk, fdbkk, wheelk, N)
                    combfits[N].append(currfit)
            cellfits[comb] = combfits
        fits[cell] = cellfits

    fw = open('rcos_synthetic_data_multifits.p', 'wb')
    pickle.dump(fits, fw)
    fw.close()
