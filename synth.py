import numpy as np
import pandas as pd
from numpy.random import uniform, normal
from scipy.interpolate import interp1d
from brainbox.modeling.glm import convbasis
import brainbox.modeling.glm
import brainbox.modeling.glm_linear as linglm

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


def simulate_cell(stimkern, fdbkkern, pgain, gain, num_trials=500, linear=False):
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
        if not linear:
            ratevals = np.exp(kernsum + gain) * BINSIZE
            spikecounts = np.random.poisson(ratevals)
        else:
            ratevals = (kernsum + gain) * BINSIZE
            ratevals[ratevals < 0] = 0
            spikecounts = np.random.poisson(ratevals)
        spike_times = []
        noisevals = uniform(low=0, high=BINSIZE - 1e-8, size=np.max(spikecounts))
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


def concat_simcell_data(trialspikes, contrasts, priors, stimtimes, fdbktimes):
    trialsdf = pd.DataFrame()
    trialends = np.cumsum(fdbktimes + 0.8)
    trialends = np.pad(trialends, ((1, 0)), constant_values=0)
    cat_stimtimes = np.array([trialends[i] + st for i, st in enumerate(stimtimes)])
    cat_fdbktimes = np.array([trialends[i] + ft for i, ft in enumerate(fdbktimes)])
    trialsdf['contrasts'] = contrasts
    trialsdf['priors'] = priors
    trialsdf['trial_start'] = trialends[:-1]
    trialsdf['trial_end'] = trialends[1:]
    trialsdf['stimOn_times'] = cat_stimtimes
    trialsdf['feedback_times'] = cat_fdbktimes

    indices = trialsdf.index
    adj_spkt = np.hstack([trialsdf.loc[i].trial_start + np.array(t)
                          for i, t in zip(indices, trialspikes)])    
    return adj_spkt, trialsdf


def simulate_cell_realdf(trialsdf, stimkern, fdbkkern, wheelkern, pgain, gain,
                         numtrials=None, linear=False):
    trialspikes = []
    if numtrials is not None:
        if numtrials > len(trialsdf.index):
            raise ValueError('numtrials must be less than number of trialsdf rows')
        keeptrials = np.random.choice(trialsdf.index, numtrials, replace=False)
    else:
        keeptrials = trialsdf.index[:-1]
    for indx in keeptrials:
        tr = trialsdf.loc[indx]
        start = t_b
        end = tr.feedback_times - tr.trial_start
        if hasattr(tr, 'bias'):
            prior = tr.bias
            newpr = trialsdf.loc[indx + 1].bias
        else:
            prior = 1
            newpr = 1
        wheel = trialsdf.loc[indx].wheel_velocity
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
                                 offset=-np.ceil(0.4 / BINSIZE).astype(int)).flatten()
        else:
            wheelarr = np.ones_like(stimarr) * 1e-8
        fdbkind = int(np.ceil(end / BINSIZE))
        priorarr = np.array([prior] * fdbkind + [newpr] * (trial_len - fdbkind))
        priorarr = pgain * priorarr
        kernsum = priorarr + stimarr + fdbkarr + wheelarr
        # ratevals = np.exp(kernsum + gain) * BINSIZE
        if not linear:
            ratevals = (np.exp(kernsum + gain)) * BINSIZE
            spikecounts = np.random.poisson(ratevals)
        else:
            ratevals = (kernsum + gain) * BINSIZE
            ratevals[ratevals < 0] = 0
            spikecounts = np.round(np.random.normal(loc=ratevals, scale=gain)).astype(int)
            # spikecounts = np.random.poisson(ratevals)
            # ratevals[ratevals < 0] = 0

        spike_times = []
        noisevals = uniform(low=0, high=BINSIZE - 1e-8, size=np.max(spikecounts))
        for i in np.nonzero(spikecounts)[0]:
            curr_t = i * BINSIZE
            for j in range(spikecounts[i]):
                jitterspike = curr_t + noisevals[j]
                if jitterspike < 0:
                    jitterspike = 0
                spike_times.append(jitterspike)
        trialspikes.append(spike_times)
    return trialspikes, keeptrials


def stepfunc(row):
    def binf(t):
        return np.ceil(t / 0.02).astype(int)
    currvec = np.ones(binf(row.stimOn_times)) * row.bias
    nextvec = np.ones(binf(row.duration) - binf(row.stimOn_times)) * row.bias_next
    return np.hstack((currvec, nextvec))


def fit_sim(trialsdf, stimkern, fdbkkern, wheelkern, ntrials, priorgain=0, gain=2.5, retglm=False,
            linear=False, ret_spikes=False, method='pure'):
    fitbool = [test is not None for test in (stimkern, fdbkkern, wheelkern)]
    if all(np.invert(fitbool)):
        raise TypeError('All kernels cannot be None')
    if stimkern is None:
        stimkern = np.zeros(int(0.6 / BINSIZE))
    if fdbkkern is None:
        fdbkkern = np.zeros(int(0.6 / BINSIZE))
    if wheelkern is None:
        wheelkern = np.zeros(int(0.4 / BINSIZE))
    trialspikes, indices = simulate_cell_realdf(trialsdf,
                                                stimkern,
                                                fdbkkern,
                                                wheelkern,
                                                pgain=priorgain, gain=gain,
                                                numtrials=ntrials, linear=linear)
    adj_spkt = np.hstack([trialsdf.loc[i].trial_start + np.array(t)
                          for i, t in zip(indices, trialspikes)])
    sess_trialspikes = np.sort(adj_spkt)
    sess_clu = np.ones_like(adj_spkt, dtype=int)
    biascols = ['bias', 'bias_next'] if hasattr(trialsdf, 'bias') else []
    fitdf = trialsdf.loc[indices][['trial_start', 'trial_end',
                                   'stimOn_times', 'feedback_times',
                                   'wheel_velocity', *biascols]].sort_index()

    nglm = linglm.LinearGLM(fitdf, sess_trialspikes, sess_clu,
                            {'trial_start': 'timing',
                             'stimOn_times': 'timing',
                             'feedback_times': 'timing',
                             'trial_end': 'timing',
                             'wheel_velocity': 'continuous',
                             'bias': 'value',
                             'bias_next': 'value'},
                            mintrials=1, train=0.7)
    bases = glm.full_rcos(0.4, 15, nglm.binf)
    longbases = glm.full_rcos(0.6, 15, nglm.binf)
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
    nglm.fit(method=method, alpha=0)
    combined_weights = nglm.combine_weights()
    retlist = []
    retlist.append(nglm.intercepts.iloc[0])
    if fitbool[0] is True:
        retlist.append(combined_weights['stim'].loc[1])
    if fitbool[1] is True:
        retlist.append(combined_weights['fdbk'].loc[1])
    if fitbool[2] is True:
        retlist.append(combined_weights['wheel'].loc[1])
    retlist.append(nglm.score().loc[1])
    if retglm:
        retlist.append(nglm)
    if ret_spikes:
        retlist.append(adj_spkt)
    return retlist


def kerngen(length, ngauss, sign=1):
    tbins = np.ceil(length / BINSIZE).astype(int)
    kernarr = np.zeros((tbins, ngauss))
    for i in range(ngauss):
        height = np.random.choice(np.linspace(1, 2.5, 5))
        # sign = np.random.choice([-1, 1]) if i != 0 else 1
        center = np.random.choice(np.arange(int(tbins / 6), int(5 * tbins / 6), 50))
        spread = np.random.choice([tbins / 64, tbins / 32, tbins / 16])
        shape = np.exp(-0.5 * ((np.arange(tbins) - center) / (spread ** 2))**2)
        kernarr[:, i] = sign * height * shape
    return np.sum(kernarr, axis=1)


if __name__ == "__main__":
    import pickle
    # import pandas as pd
    import itertools as it
    from tqdm import tqdm
    # from brainbox.singlecell import calculate_peths
    from oneibl import one
    from export_funs import trialinfo_to_df
    # from prior_funcs import fit_sess_psytrack
    from brainbox.modeling import glm

    linear = True

    one = one.ONE()
    subject = 'ZM_2240'
    sessdate = '2020-01-22'
    ids = one.search(subject=subject, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.times'])
    trialsdf = trialinfo_to_df(ids[0], maxlen=2., ret_wheel=True)
    # wts, stds = fit_sess_psytrack(ids[0], maxlength=2., as_df=True)
    # trialsdf = pd.concat((trialsdf, wts['bias']), axis=1)
    # trialsdf = trialsdf[np.isfinite(trialsdf.bias)]
    # bias_next = np.roll(trialsdf['bias'], -1)
    # bias_next = pd.Series(bias_next, index=trialsdf['bias'].index)[:-1]
    # trialsdf['bias_next'] = bias_next

    nvals = np.linspace(100, len(trialsdf) - 5, 3, dtype=int)
    gain = np.log(10)

    cell_ids = list(range(5))
    kernelcombs = list(it.product(*[(False, True)] * 3))  # Boolean combination of kernels
    _ = kernelcombs.pop(0)
    fits = {}
    for cell in tqdm(cell_ids, desc='Cell'):
        cellfits = {}
        left_t = trialsdf[np.isfinite(trialsdf.contrastLeft)].stimOn_times
        fdbk = trialsdf[np.isfinite(trialsdf.contrastLeft)].feedback_times
        # Generate random kernels for stim and fdbk with random # of gaussians
        while True:
            stimkern = kerngen(0.6, np.random.choice([1, 2, 3]))
            fdbkkern = kerngen(0.6, np.random.choice([1, 2, 3]), sign=-1)
            wheelkern = np.exp(-0.5 * ((np.linspace(0, 0.4, int(0.4 / BINSIZE)) - 0.3) / 0.05)**2)
            wheelkern = wheelkern / np.max(wheelkern) * np.random.choice(np.arange(1.5, 6))
            keeparrs = [x for x in (stimkern, fdbkkern, wheelkern) if x is not None]
            peaksum = np.sum([x.max() for x in keeparrs])
            if (peaksum > 4) & (peaksum < 7):
                break

        if linear:
            stimkern, fdbkkern, wheelkern = stimkern * 4, fdbkkern * 4, wheelkern * 10

        cellfits['kernels'] = (stimkern, fdbkkern, wheelkern)
        for comb in tqdm(kernelcombs, desc='Kernel combination', leave=False):
            combfits = {}
            for N in tqdm(nvals, desc='N values', leave=False):
                combfits[N] = []
                for i in tqdm(range(1), desc='iter', leave=False):
                    stimk = stimkern if comb[0] else None
                    fdbkk = fdbkkern if comb[1] else None
                    wheelk = wheelkern if comb[2] else None
                    currfit = fit_sim(trialsdf, stimk, fdbkk, wheelk, N, gain=gain, linear=linear,
                                      retglm=True)
                    combfits[N].append(currfit)
            cellfits[comb] = combfits
        fits[cell] = cellfits

