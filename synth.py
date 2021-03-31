import numpy as np
import pandas as pd
from numpy.random import uniform, normal
from scipy.interpolate import interp1d
from brainbox.modeling.glm import convbasis
import brainbox.modeling.glm_linear as linglm
import brainbox.modeling.glm as glm

BINSIZE = 0.02
KERNLEN = 0.6
SHORT_KL = 0.4
NBASES = 10
rt_vals = np.array([0.20748797, 0.39415191, 0.58081585, 0.76747979, 0.95414373,
                    1.14080767, 1.32747161, 1.51413555, 1.70079949, 1.88746343])
rt_probs = np.array([0.15970962, 0.50635209, 0.18693285, 0.0707804, 0.02540835,
                     0.01633394, 0.00907441, 0.00725953, 0.00544465, 0.01270417])
contrastvals = [0, 0.0625, 0.125, 0.25, 1.] + [-0.0625, -0.125, -0.25, -1.]
longbases = glm.full_rcos(KERNLEN, NBASES, lambda x: np.ceil(x / BINSIZE).astype(int))
shortbases = glm.full_rcos(SHORT_KL, NBASES, lambda x: np.ceil(x / BINSIZE).astype(int))


def _generate_pseudo_blocks(n_trials, factor=60, min_=20, max_=100):
    block_ids = []
    while len(block_ids) < n_trials:
        x = np.random.exponential(factor)
        while (x <= min_) | (x >= max_):
            x = np.random.exponential(factor)
        if (len(block_ids) == 0) & (np.random.randint(2) == 0):
            block_ids += [0] * int(x)
        elif (len(block_ids) == 0):
            block_ids += [1] * int(x)
        elif block_ids[-1] == 0:
            block_ids += [1] * int(x)
        elif block_ids[-1] == 1:
            block_ids += [0] * int(x)
    return np.array(block_ids[:n_trials])


def kerngen(length):
    if length not in (KERNLEN, SHORT_KL):
        raise ValueError(f'length must be {KERNLEN} or {SHORT_KL}')
    rng = np.random.default_rng()
    weights = rng.uniform(low=-2, high=2, size=NBASES)
    bases = longbases if length == KERNLEN else shortbases
    return bases @ weights


def simulate_cell(stimkerns, fdbkkerns, wheelkern, pgain, gain, wheeltraces,
                  num_trials=500, linear=False, ret_raw=False):
    stimtimes = np.ones(num_trials) * 0.4
    fdbktimes = np.random.choice(rt_vals, size=num_trials, p=rt_probs) \
        + stimtimes + normal(size=num_trials) * 0.05
    priorbool = _generate_pseudo_blocks(num_trials)
    priors = np.array([{1: 0.2, 0: 0.8}[x] for x in priorbool])
    contrasts = np.random.choice(contrastvals, replace=True, size=num_trials)
    feedbacktypes = np.random.choice([-1, 1], size=num_trials, p=[0.1, 0.9])
    wheelmoves = np.random.choice(np.arange(len(wheeltraces)), size=num_trials)
    trialspikes = []
    trialrates = []
    trialwheel = []
    if ret_raw:
        trialcont = []
    trialrange = range(num_trials)
    zipiter = zip(trialrange, stimtimes, fdbktimes, priors,
                  contrasts, feedbacktypes, wheelmoves)
    for i, start, end, prior, contrast, feedbacktype, wheelchoice in zipiter:
        if i == (len(priors) - 1):
            continue
        trial_len = int(np.ceil((end + KERNLEN) / BINSIZE))
        stimarr = np.zeros(trial_len)
        fdbkarr = np.zeros(trial_len)
        stimarr[int(np.ceil(start / BINSIZE))] = 1
        fdbkarr[int(np.ceil(end / BINSIZE))] = 1
        stimkern = stimkerns[0] if contrast > 0 else stimkerns[1]
        fdbkkern = fdbkkerns[0] if feedbacktype == 1 else fdbkkerns[1]
        stimarr = np.convolve(stimkern, stimarr)[:trial_len]
        fdbkarr = np.convolve(fdbkkern, fdbkarr)[:trial_len]
        fdbkind = int(np.ceil(end / BINSIZE))

        wheel = wheeltraces[wheelchoice].copy()
        lendiff = int(np.ceil((end + KERNLEN) / BINSIZE)) - wheel.shape[0]
        if lendiff >= 0:
            wheel = np.pad(wheel, (0, lendiff), constant_values=0)
        else:
            wheel = wheel[:lendiff]
        wheelinterp = interp1d(np.arange(len(wheel)) * BINSIZE,
                               wheel, fill_value='extrapolate')
        wheelnew = wheelinterp(np.arange(trial_len) * BINSIZE)
        wheelarr = convbasis(wheelnew.reshape(-1, 1),
                             wheelkern.reshape(-1, 1),
                             offset=-np.ceil(SHORT_KL / BINSIZE).astype(int)).flatten()

        priorarr = np.array([prior] * fdbkind +
                            [priors[i + 1]] * (trial_len - fdbkind))
        priorarr = pgain * priorarr
        kernsum = priorarr + stimarr + fdbkarr + wheelarr
        if not linear:
            ratevals = np.exp(kernsum + gain) * BINSIZE
            spikecounts = np.random.poisson(ratevals)
        else:
            ratevals = (kernsum + gain) * BINSIZE
            # ratevals[ratevals < 0] = 0
            contspikecounts = np.random.normal(
                loc=ratevals, scale=gain * BINSIZE)
            spikecounts = np.round(contspikecounts).astype(int)
            # spikecounts[spikecounts < 0] = 0
            # spikecounts = spikecounts.astype(int)
            # spikecounts = np.random.poisson(ratevals)
        if ret_raw:
            trialcont.append(contspikecounts)
        spike_times = []

        noisevals = uniform(low=0, high=BINSIZE - 1e-8,
                            size=np.max(spikecounts))
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
        trialrates.append(ratevals)
        trialwheel.append(wheel)
    retlist = [trialspikes, contrasts, priors, stimtimes,
               fdbktimes, feedbacktypes, trialwheel, trialrates,
               trialcont if ret_raw else None]
    return retlist


def concat_simcell_data(trialspikes, contrasts, priors, stimtimes, fdbktimes, feedbacktypes,
                        trialwheel):
    trialsdf = pd.DataFrame()
    trialends = np.cumsum(fdbktimes + KERNLEN)
    trialends = np.pad(trialends, ((1, 0)), constant_values=0)
    cat_stimtimes = np.array(
        [trialends[i] + st for i, st in enumerate(stimtimes)])
    cat_fdbktimes = np.array(
        [trialends[i] + ft for i, ft in enumerate(fdbktimes)])
    trialsdf['contrasts'] = contrasts
    trialsdf['bias'] = priors
    trialsdf['bias_next'] = np.pad(priors[1:], (0, 1), constant_values=0)
    trialsdf['trial_start'] = trialends[:-1]
    trialsdf['trial_end'] = trialends[1:]
    trialsdf['stimOn_times'] = cat_stimtimes
    trialsdf['feedback_times'] = cat_fdbktimes
    trialsdf['feedback_type'] = feedbacktypes
    trialwheel.append(0)
    trialsdf['wheel_velocity'] = trialwheel

    indices = trialsdf.index
    adj_spkt = np.hstack([trialsdf.loc[i].trial_start + np.array(t)
                          for i, t in zip(indices, trialspikes)])
    return adj_spkt, trialsdf.iloc[:-1]


def stepfunc(row):
    def binf(t):
        return np.ceil(t / BINSIZE).astype(int)
    currvec = np.ones(binf(row.stimOn_times)) * row.bias
    nextvec = np.ones(binf(row.duration) -
                      binf(row.stimOn_times)) * row.bias_next
    return np.hstack((currvec, nextvec))


def fit_full_sim(trialsdf, stimkerns, fdbkkerns, wheelkern, wheeltraces, ntrials,
                 priorgain=0, gain=2.5, retglm=False, linear=False, use_raw=False):
    if wheelkern is None:
        wheelkern = np.zeros(int(SHORT_KL / BINSIZE))
    out = simulate_cell(stimkerns, fdbkkerns, wheelkern, wheeltraces=wheeltraces,
                        pgain=priorgain, gain=gain, num_trials=ntrials, linear=linear,
                        ret_raw=use_raw)
    if use_raw:
        rawoutput = np.hstack(out[-1])
    trialspikes, contrasts, priors, stimtimes, fdbktimes, feedbacktypes, trialwheel = out[:-2]
    adj_spkt, trialsdf = concat_simcell_data(trialspikes, contrasts, priors, stimtimes, fdbktimes,
                                             feedbacktypes, trialwheel)
    sess_trialspikes = np.sort(adj_spkt)
    sess_clu = np.ones_like(adj_spkt, dtype=int)

    nglm = linglm.LinearGLM(trialsdf, sess_trialspikes, sess_clu,
                            {'trial_start': 'timing',
                             'stimOn_times': 'timing',
                             'feedback_times': 'timing',
                             'trial_end': 'timing',
                             'contrasts': 'value',
                             'feedback_type': 'value',
                             'wheel_velocity': 'continuous',
                             'bias': 'value',
                             'bias_next': 'value'},
                            mintrials=1, train=0.7)
    bases = glm.full_rcos(SHORT_KL, NBASES, nglm.binf)
    longbases = glm.full_rcos(KERNLEN, NBASES, nglm.binf)
    nglm.add_covariate_timing('stimL', 'stimOn_times', longbases,
                              cond=lambda tr: tr.contrasts > 0,
                              desc='synth stimon')
    nglm.add_covariate_timing('stimR', 'stimOn_times', longbases,
                              cond=lambda tr: tr.contrasts <= 0,
                              desc='synth stimon')
    nglm.add_covariate_timing('correct', 'feedback_times', longbases,
                              cond=lambda tr: tr.feedback_type == 1, desc='synth fdbk')
    nglm.add_covariate_timing('incorrect', 'feedback_times', longbases,
                              cond=lambda tr: tr.feedback_type == -1, desc='synth fdbk')
    nglm.add_covariate('wheel', trialsdf['wheel_velocity'], bases, offset=-SHORT_KL,
                       desc='synthetic wheel move')
    if priorgain != 0:
        nglm.add_covariate_raw(
            'prior', stepfunc, desc='Step function on prior estimate')

    nglm.compile_design_matrix()
    _, s, v = np.linalg.svd(
        nglm.dm[:, nglm.covar['wheel']['dmcol_idx']], full_matrices=False)
    variances = s**2 / (s**2).sum()
    n_keep = np.argwhere(np.cumsum(variances) >= 0.99999)[0, 0]
    wheelcols = nglm.dm[:, nglm.covar['wheel']['dmcol_idx']]
    reduced = wheelcols @ v[:n_keep].T
    bases_reduced = bases @ v[:n_keep].T
    keepcols = ~np.isin(
        np.arange(nglm.dm.shape[1]), nglm.covar['wheel']['dmcol_idx'])
    basedm = nglm.dm[:, keepcols]
    nglm.dm = np.hstack([basedm, reduced])
    nglm.covar['wheel']['dmcol_idx'] = nglm.covar['wheel']['dmcol_idx'][:n_keep]
    nglm.covar['wheel']['bases'] = bases_reduced
    if np.linalg.cond(nglm.dm) > 1e6:
        print('Bad COND!')
        # return None
    if use_raw:
        nglm.binnedspikes = rawoutput.reshape(-1, 1)
    nglm.fit(method='ridge', alpha=0)
    combined_weights = nglm.combine_weights()
    retlist = []
    retlist.append(nglm.intercepts.iloc[0])
    retlist.append(combined_weights['stimL'].loc[1])
    retlist.append(combined_weights['stimR'].loc[1])
    retlist.append(combined_weights['correct'].loc[1])
    retlist.append(combined_weights['incorrect'].loc[1])
    retlist.append(combined_weights['wheel'].loc[1])
    if priorgain != 0:
        retlist.append(combined_weights['prior'])
    retlist.append(nglm.score().loc[1])
    if retglm:
        retlist.append(nglm)
    return retlist


if __name__ == "__main__":
    import itertools as it
    from tqdm import tqdm
    from oneibl import one
    from export_funs import trialinfo_to_df
    import matplotlib.pyplot as plt
    from brainbox.modeling import glm

    linear = True
    rawobservations = True

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
    wheeltraces = trialsdf.wheel_velocity.to_list()

    cell_ids = list(range(10))
    # Boolean combination of kernels
    kernelcombs = list(it.product(*[(False, True)] * 3))
    _ = kernelcombs.pop(0)
    fits = {}
    for cell in tqdm(cell_ids, desc='Cell'):
        cellfits = {}
        left_t = trialsdf[np.isfinite(trialsdf.contrastLeft)].stimOn_times
        fdbk = trialsdf[np.isfinite(trialsdf.contrastLeft)].feedback_times
        # Generate random kernels for stim and fdbk with random # of gaussians
        stimkernL = kerngen(0.6)
        stimkernR = kerngen(0.6)
        fdbkkern1 = kerngen(0.6)
        fdbkkern2 = kerngen(0.6)
        wheelkern = kerngen(0.4)

        if linear:
            stimkernL *= 4
            stimkernR *= 4
            fdbkkern1 *= 4
            fdbkkern2 *= 4
            wheelkern *= 4
        
        fits[cell] = {}
        fits[cell]['kernels'] = [
            (stimkernL, stimkernR), (fdbkkern1, fdbkkern2), wheelkern]
        for N in [400, 600, 10000]:
            fits[cell][N] = fit_full_sim(trialsdf, (stimkernL, stimkernR),
                                         (fdbkkern1, fdbkkern2),
                                         wheelkern,
                                         wheeltraces,
                                         gain=15,
                                         ntrials=N, linear=linear, retglm=False,
                                         use_raw=rawobservations)

    for cell in fits:
        fig, ax = plt.subplots(3, 2, figsize=(8, 8))
        ax = ax.flatten()
        ax[-1].set_visible(False)
        bigN = fits[cell][10000]
        middleN = fits[cell][600]
        littleN = fits[cell][400]
        kerns = fits[cell]['kernels']
        # StimL
        ax[0].plot(np.arange(0, 0.6, BINSIZE), kerns[0]
                   [0], label='generative kernel')
        ax[0].plot(littleN[1] / 0.02, label='400 trials')
        ax[0].plot(middleN[1] / 0.02, label='600 trials')
        ax[0].plot(bigN[1] / 0.02, label='10000 trials')
        ax[0].legend()
        ax[0].set_title('Left stim kernel')
        # StimR
        ax[1].plot(np.arange(0, 0.6, BINSIZE), kerns[0]
                   [1], label='generative kernel')
        ax[1].plot(littleN[2] / 0.02, label='400 trials')
        ax[1].plot(middleN[2] / 0.02, label='600 trials')
        ax[1].plot(bigN[2] / 0.02, label='10000 trials')
        ax[1].legend()
        ax[1].set_title('Right stim kernel')
        # Correct
        ax[2].plot(np.arange(0, 0.6, BINSIZE), kerns[1]
                   [0], label='generative kernel')
        ax[2].plot(littleN[3] / 0.02, label='400 trials')
        ax[2].plot(middleN[3] / 0.02, label='600 trials')
        ax[2].plot(bigN[3] / 0.02, label='10000 trials')
        ax[2].legend()
        ax[2].set_title('Correct kernel')
        # Incorrect
        ax[3].plot(np.arange(0, 0.6, BINSIZE), kerns[1]
                   [1], label='generative kernel')
        ax[3].plot(littleN[4] / 0.02, label='400 trials')
        ax[3].plot(middleN[4] / 0.02, label='600 trials')
        ax[3].plot(bigN[4] / 0.02, label='10000 trials')
        ax[3].legend()
        ax[3].set_title('Incorrect kernel')
        # Wheel
        ax[4].plot(np.arange(-0.4, 0, BINSIZE),
                   kerns[2], label='generative kernel')
        ax[4].plot(littleN[5] / 0.02, label='400 trials')
        ax[4].plot(middleN[5] / 0.02, label='600 trials')
        ax[4].plot(bigN[5] / 0.02, label='10000 trials')
        ax[4].legend()
        ax[4].set_title('Wheel kernel')
        plt.suptitle(f'Synthetic unit {cell}')
        plt.tight_layout()
        rawstr = '_raw' if rawobservations else ''
        plt.savefig(f'/home/berk/Documents/psth-plots/synth/kernels/cell{cell}_'
                    f'fits_ridge{rawstr}.png',
                    dpi=400)

# %%
