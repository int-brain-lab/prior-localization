import numpy as np
from ibllib.atlas import BrainRegions
from brainwidemap.bwm_loading import load_good_units, merge_probes
from brainbox.population.decode import get_spike_counts_in_bins


region_defaults = {
    'custom': ['VISp', 'MOs'],
    'widefield': [
            ["ACAd"],
            ["AUDd"],
            ["AUDp"],
            ["AUDpo"],
            ["AUDv"],
            ["FRP"],
            ["MOB"],
            ["MOp"],
            ["MOs"],
            ["PL"],
            ["RSPagl"],
            ["RSPd"],
            ["RSPv"],
            ["SSp-bfd"],
            ["SSp-ll"],
            ["SSp-m"],
            ["SSp-n"],
            ["SSp-tr"],
            ["SSp-ul"],
            ["SSp-un"],
            ["SSs"],
            ["TEa"],
            ["VISa"],
            ["VISal"],
            ["VISam"],
            ["VISl"],
            ["VISli"],
            ["VISp"],
            ["VISpl"],
            ["VISpm"],
            ["VISpor"],
            ["VISrl"],
        ]
}


def stage_ephys(one, session_id, probe_id):
    pass


def stage_widefield(one, session_id, probe_id):
    pass


def prepare_ephys(one, session_id, probe_name, intervals, qc=1, regions='default', min_units=10):

    # Load spikes and clusters and potentially merge probes
    if isinstance(probe_name, list) and len(probe_name) > 1:
        to_merge = [load_good_units(one, pid=None, eid=session_id, qc=qc, pname=probe_name)
                    for probe_name in probe_name]
        spikes, clusters = merge_probes([spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge])
    else:
        spikes, clusters = load_good_units(one, pid=None, eid=session_id, qc=qc, pname=probe_name)

    # Prepare list of brain regions
    brainreg = BrainRegions()
    beryl_regions = brainreg.acronym2acronym(clusters['acronym'], mapping="Beryl")
    if isinstance(regions, str):
        if regions in region_defaults.keys():
            regions = region_defaults[regions]
        elif regions == 'single_regions':
            regions = [[k] for k in np.unique(beryl_regions) if k not in ['root', 'void']]
        elif regions == 'all_regions':
            regions = [np.unique([r for r in beryl_regions if r not in ['root', 'void']])]
        else:
            regions = [regions]
    elif isinstance(regions, list):
        pass

    binned_spikes = []
    n_units = []
    for region in regions:
        # find all clusters in region (where region can be a list of regions)
        region_mask = np.isin(beryl_regions, region)
        if sum(region_mask) < min_units:
            print(region, f"{region} below min units threshold : {sum(region_mask)}, not decoding")
            continue
        # find all spikes in those clusters
        spike_mask = spikes['clusters'].isin(clusters[region_mask]['cluster_id'])
        spikes['times'][spike_mask]
        spikes['clusters'][spike_mask]
        n_units.append(sum(region_mask))
        binned, _ = get_spike_counts_in_bins(spikes['times'], spikes['clusters'], intervals)
        binned = binned.T  # binned is a 2D array
        binned_spikes.append([x[None, :] for x in binned])
    return binned_spikes, regions, n_units


#
# def prepare_widefield():
#     from prior_localization.decoding.functions.process_inputs import get_bery_reg_wfi
#
#     beryl_reg = get_bery_reg_wfi(neural_dict, **kwargs)
#     reg_mask = select_widefield_imaging_regions(neural_dict, region, **kwargs)
#     msub_binned = preprocess_widefield_imaging(neural_dict, reg_mask, **kwargs)
#     n_units = np.sum(reg_mask)
#
#
#     return hemisphere

