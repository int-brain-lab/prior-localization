from synth import simulate_cell
import numpy as np
import pickle
import pandas as pd
from scipy.io import savemat
from iofuns import loadmat
from tqdm import tqdm
from brainbox.singlecell import calculate_peths
from oneibl import one
from export_funs import trialinfo_to_df
from prior_funcs import fit_sess_psytrack
from brainbox.modeling import glm
import os

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
left_t = trialsdf[np.isfinite(trialsdf.contrastLeft)].stimOn_times
fdbk = trialsdf[np.isfinite(trialsdf.contrastLeft)].feedback_times

for cell in [0, 1, 2, 3, 4]:
    left_rate, _ = calculate_peths(spk_times, spk_clu, [cell], left_t, pre_time=0,
                                   post_time=0.6, bin_size=0.005)
    left_rate.means = left_rate.means - left_rate.means.min()
    left_rate = left_rate.means.flatten()
    fdbk_rate = np.zeros_like(left_rate)

    trialspikes, _, _, stimtimes, fdbktimes = simulate_cell(left_rate, fdbk_rate, 0, 0, 1000)

    outdict = {'spk_times': trialspikes, 'stimt': stimtimes, 'fdbkt': fdbktimes}
    savemat('synthspikes.mat', outdict)

    adj_spk = []
    trdict = []
    start = 0
    for i in range(len(trialspikes)):
        currtr = {} 
        end = fdbktimes[i] + 0.6
        currtr['trial_start'] = start
        currtr['trial_end'] = end + start
        currtr['stimOn_times'] = start + 0.4
        currtr['feedback_times'] = start + fdbktimes[i]
        for t in trialspikes[i]:
            adj_spk.append(t + start)
        start += end
        trdict.append(currtr)

    syn_spk = np.array(adj_spk)
    fitdf = pd.DataFrame(trdict)
    nglm = glm.NeuralGLM(fitdf, syn_spk, np.ones_like(syn_spk, dtype=int),
                        {'trial_start': 'timing',
                        'trial_end': 'timing',
                        'stimOn_times': 'timing',
                        'feedback_times': 'timing'}, train=1.)
    bases = glm.raised_cosine(0.6, 10, nglm.binf)
    nglm.add_covariate_timing('stim', 'stimOn_times', bases)
    nglm.compile_design_matrix()
    nglm.bin_spike_trains()
    # nglm.dm = np.roll(nglm.dm, -1, axis=0)
    nglm.fit('minimize')
    combined_weighst = nglm.combine_weights()
    pydm = nglm.dm
    pybinspk = nglm.binnedspikes

    os.system('matlab -nodisplay -r fitsynthdata; exit;')
    matfile = loadmat('synthmatfit.mat')
    matdm = matfile['savedm']
    plt.figure()
    plt.pcolor(pydm - matdm[:, 1:])
    matbinspk = matfile['y'].toarray()
    matkerns = matfile['weights']
    plt.figure()
    plt.plot(matkerns['stim']['tr'], matkerns['stim']['data'], label='MATLAB')
    plt.plot(combined_weighst['stim'].loc[1], label='Python')
