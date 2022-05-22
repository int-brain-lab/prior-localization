import os
import numpy as np
import pandas as pd
from pathlib import Path
from ibllib.atlas import BrainRegions
from tqdm import tqdm
import pickle
from braindelphi.decoding.settings import modeldispatcher
from behavior_models.models.utils import build_path as build_path_mut

def check_bhv_fit_exists(subject, model, eids, resultpath):
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
    return mask & (trialsdf.choice != 0)

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


# %% Define helper functions for dask workers to use
def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, N, output_path,
                        time_window, today, target, add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([
        today, region, 'target', target, 'timeWindow',
        str(start_tw).replace('.', '_'),
        str(end_tw).replace('.', '_'), 'pseudo_id',
        str(pseudo_id), add_to_saving_path
    ]) + '.pkl'
    for folder in [subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    outdict = {
        'fit': fit_result,
        'pseudo_id': pseudo_id,
        'subject': subject,
        'eid': eid,
        'probe': probe,
        'region': region,
        'N_units': N
    }
    fw = open(probefolder.joinpath(fn), 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


def return_path(eid, sessdf, pseudo_ids=[-1], **kwargs):
    """
    Parameters
    ----------
    single_region: Bool, decoding using region wise or pulled over regions
    eid: eid of session
    sessdf: dataframe of session eid
    pseudo_id: whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered.
    can not be 0
    nb_runs: nb of independent runs performed. this was added after consequent variability was observed across runs.
    modelfit_path: outputs of behavioral fits
    output_path: outputs of decoding fits
    one: ONE object -- this is not to be used with dask, this option is given for debugging purposes
    """

    df_insertions = sessdf.loc[sessdf['eid'] == eid]
    subject = df_insertions['subject'].to_numpy()[0]
    brainreg = BrainRegions()

    filenames = []
    if kwargs['merged_probes']:
        across_probes = {'regions': [], 'clusters': [], 'times': [], 'qc_pass': []}
        for _, ins in df_insertions.iterrows():
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = brainreg.acronym2acronym(clusters.atlas_id, mapping='Beryl')
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            across_probes['regions'].extend(beryl_reg)
            across_probes['qc_pass'].extend(qc_pass)
        across_probes = {k: np.array(v) for k, v in across_probes.items()}
        # warnings.filterwarnings('ignore')
        if kwargs['single_region']:
            regions = [[k] for k in np.unique(across_probes['regions'])]
        else:
            regions = [np.unique(across_probes['regions'])]
        df_insertions_iterrows = pd.DataFrame.from_dict({
            '1': 'mergedProbes'
        },
                                                        orient='index',
                                                        columns=['probe']).iterrows()
    else:
        df_insertions_iterrows = df_insertions.iterrows()

    for i, ins in df_insertions_iterrows:
        probe = ins['probe']
        if not kwargs['merged_probes']:
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = brainreg.acronym2acronym(clusters.atlas_id, mapping='Beryl')
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            regions = np.unique(beryl_reg)
        for region in regions:
            if kwargs['merged_probes']:
                reg_mask = np.isin(across_probes['regions'], region)
                reg_clu_ids = np.argwhere(reg_mask & across_probes['qc_pass']).flatten()
            else:
                reg_mask = beryl_reg == region
                reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            N_units = len(reg_clu_ids)
            if N_units < kwargs['min_units']:
                continue

            for pseudo_id in pseudo_ids:
                filenames.append(
                    save_region_results(
                        None,
                        pseudo_id,
                        subject,
                        eid,
                        probe,
                        str(np.squeeze(region)) if kwargs['single_region'] else 'allRegions',
                        N_units,
                        output_path=kwargs['output_path'],
                        time_window=kwargs['time_window'],
                        today=kwargs['today'],
                        compute=False))
    return filenames


