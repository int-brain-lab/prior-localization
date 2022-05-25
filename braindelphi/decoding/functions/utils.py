import os
import numpy as np
import pandas as pd
from pathlib import Path
from ibllib.atlas import BrainRegions
from tqdm import tqdm
import pickle
from behavior_models.models.utils import build_path as build_path_mut


def check_bhv_fit_exists(subject, model, eids, resultpath, modeldispatcher):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    path_results_mouse = 'model_%s_' % modeldispatcher[model]
    trunc_eids = [eid.split('-')[0] for eid in eids]
    filen = build_path_mut(path_results_mouse, trunc_eids)
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    fullpath = subjmodpath.joinpath(filen)
    return os.path.exists(fullpath), fullpath


def compute_mask(trialsdf, **kwargs):
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf[kwargs['align_time']]
    mask = trialsdf[kwargs['align_time']].notna() & trialsdf['firstMovement_times'].notna()
    if kwargs['no_unbias']:
        mask = mask & (trialsdf.probabilityLeft != 0.5).values
    if kwargs['min_rt'] is not None:
        mask = mask & (~(trialsdf.react_times < kwargs['min_rt'])).values
    mask = mask & (trialsdf.choice != 0)
    return mask


def return_regions(eid, sessdf, QC_CRITERIA=1, NUM_UNITS=10):
    df_insertions = sessdf.loc[sessdf['eid'] == eid]
    brainreg = BrainRegions()
    my_regions = {}
    for i, ins in tqdm(df_insertions.iterrows(), desc='Probe: ', leave=False):
        probe = ins['probe']
        spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
        clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
        beryl_reg = brainreg.acronym2acronym(clusters.atlas_id, mapping='Beryl')
        qc_pass = (clusters['label'] >= QC_CRITERIA).values
        regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        probe_regions = []
        for region in tqdm(regions, desc='Region: ', leave=False):
            reg_mask = (beryl_reg == region)
            reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            if len(reg_clu_ids) > NUM_UNITS:
                probe_regions.append(region)
        my_regions[probe] = probe_regions
    return my_regions


def get_save_path(
        pseudo_id, subject, eid, probe, region, output_path, time_window, today, target,
        add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([today, region, 'target', target,
                   'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_'),
                   'pseudo_id', str(pseudo_id), add_to_saving_path]) + '.pkl'
    save_path = probefolder.joinpath(fn)
    return save_path


def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, N, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    outdict = {
        'fit': fit_result, 'pseudo_id': pseudo_id, 'subject': subject, 'eid': eid, 'probe': probe,
        'region': region, 'N_units': N
    }
    fw = open(save_path, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return save_path
