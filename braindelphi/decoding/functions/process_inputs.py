
def preprocess_ephys():
    spikemask = np.isin(spikes.clusters, reg_clu_ids)
    regspikes = spikes.times[spikemask]
    regclu = spikes.clusters[spikemask]
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

def select_ephys_regions():
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
    return reg_clu_ids