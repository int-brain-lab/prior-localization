import numpy as np
from brainbox.population.decode import get_spike_counts_in_bins


def preprocess_ephys(reg_clu_ids, regressors, trials_df, **kwargs):
    intervals = np.vstack([
        trials_df[kwargs['align_time']] + kwargs['time_window'][0],
        trials_df[kwargs['align_time']] + kwargs['time_window'][1]
    ]).T
    spikemask = np.isin(regressors['spk_clu'], reg_clu_ids)
    regspikes = regressors['spk_times'][spikemask]
    regclu = regressors['spk_clu'][spikemask]
    binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
    return binned

def proprocess_widefield_imaging():
    frames_idx = wideFieldImaging_dict['timings'][kwargs['align_time']].values
    frames_idx = np.sort(
        frames_idx[:, None] +
        np.arange(0, kwargs['wfi_nb_frames'], np.sign(kwargs['wfi_nb_frames'])),
        axis=1,
    )
    binned = np.take(wideFieldImaging_dict['activity'][:, reg_mask],
                     frames_idx,
                     axis=0)
    binned = binned.reshape(binned.shape[0], -1).T
    return binned

def select_widefield_imaging_regions():
    region_labels = []
    reg_lab = wideFieldImaging_dict['atlas'][wideFieldImaging_dict['atlas'].acronym ==
                                             region].label.values.squeeze()
    if 'left' in kwargs['wfi_hemispheres']:
        region_labels.append(reg_lab)
    if 'right' in kwargs['wfi_hemispheres']:
        region_labels.append(-reg_lab)

    reg_mask = np.isin(wideFieldImaging_dict['clu_regions'], region_labels)
    reg_clu_ids = np.argwhere(reg_mask)
    return reg_clu_ids

def select_ephys_regions(regressors, beryl_reg, region, **kwargs):
    qc_pass = (regressors['clu_qc'] >= kwargs['qc_criteria']).values
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
    return reg_clu_ids