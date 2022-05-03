import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm

import brainbox.io.one as bbone
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_session
from ibllib.atlas import BrainRegions
import one.alf.io as alfio
from one.api import One

import sys
sys.path.append('/home/mattw/Dropbox/github/int-brain-lab/prior-localization/brainwide/decoding')
import functions.utils as dut
import functions.utils_continuous as dutc


def fit_eid(eid, bwmdf, pseudo_ids=[-1], sessiondf=None, wideFieldImaging_dict=None, **kwargs):
    """

    Parameters
    ----------
    eid : str
        eid of session
    bwmdf : pandas.DataFrame
        dataframe of bwm session
    pseudo_ids : list
        whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered;
        cannot be 0
    sessiondf : pandas.DataFrame
        the behavioral and neural dataframe when you want to bypass the bwm encoding phase
    wideFieldImaging_dict : dict
    kwargs
        target : str
        align_time : str
        time_window : tuple
            (window_start, window_end) relative to align_time
        binsize : float
        n_bins_lag : int
        estimator : sklearn.linear_model object
        hyperparameter_grid : dict
        n_pseudo : int
        n_runs : int
            number of independent runs performed. this was added after consequent variability was
            observed across runs.
        shuffle : bool
        use_imposter_session : bool
        min_units : int
        qc_criteria : float
        single_region : bool
        merged_probes : bool
        criterion : str
        min_behav_trials : int
        min_rt : float
        no_unbias : bool
        wide_field_imaging : bool
        today : str
        output_path : str
            outputs of decoding fits
        add_to_saving_path : str

    """

    # ---------------------------------------------------------------------------------------------
    # checks
    # ---------------------------------------------------------------------------------------------
    if ((wideFieldImaging_dict is None and kwargs['wide_field_imaging']) or
            (wideFieldImaging_dict is not None and not kwargs['wide_field_imaging'])):
        raise ValueError(
            'wideFieldImaging_dict must be defined for wide_field_imaging and reciprocally')

    if kwargs['wide_field_imaging'] and kwargs['wfi_nb_frames'] == 0:
        raise ValueError('wfi_nb_frames can not be 0. it is a signed non-null integer')

    if 0 in pseudo_ids:
        raise ValueError(
            'pseudo id can be -1 (actual session) or strictly greater than 0 (pseudo session)')

    one = One() if kwargs.get('one', None) is None else kwargs['one']
    brainreg = BrainRegions()

    # get session info for the selected eid
    if sessiondf is None:
        df_insertions = bwmdf.loc[bwmdf['eid'] == eid]
        subject = df_insertions['subject'].to_numpy()[0]
        if 'beh_mouseLevel_training' in kwargs.keys():
            subjeids = bwmdf.loc[bwmdf['subject'] == subject]['eid'].unique()
        else:
            subjeids = np.array([eid])
    else:
        raise NotImplementedError
        # subject = sessiondf.subject.unique()[0]
        # subjeids = sessiondf.eid.unique()

    filenames = []

    # ---------------------------------------------------------------------------------------------
    # load target data
    # ---------------------------------------------------------------------------------------------
    target_times, target_vals, skip_session = dutc.load_target_data(one, eid, kwargs['target'])
    if skip_session:
        return filenames

    # ---------------------------------------------------------------------------------------------
    # load and filter interval data
    # ---------------------------------------------------------------------------------------------
    trialsdf, skip_session = dutc.load_interval_data(
        one, eid, kwargs['align_time'], kwargs['time_window'], no_unbias=kwargs['no_unbias'],
        min_rt=kwargs['min_rt'])
    if skip_session:
        return filenames

    # ---------------------------------------------------------------------------------------------
    # split target data per trial
    # ---------------------------------------------------------------------------------------------
    target_times_list, target_val_list, trialsdf, skip_session = \
        dutc.get_target_data_per_trial_error_check(
            target_times, target_vals, trialsdf, kwargs['align_time'], kwargs['time_window'],
            kwargs['binsize'], kwargs['min_behav_trials'])
    if skip_session:
        return filenames

    # get final list of decoding intervals
    align_times = trialsdf[kwargs['align_time']].values
    interval_beg_times = align_times + kwargs['time_window'][0]
    interval_end_times = align_times + kwargs['time_window'][1]

    # ---------------------------------------------------------------------------------------------
    # select regions/insertions to decode
    # ---------------------------------------------------------------------------------------------
    if kwargs['merged_probes'] and wideFieldImaging_dict is None:
        across_probes = {'regions': [], 'clusters': [], 'times': [], 'qc_pass': []}
        for i_probe, (_, ins) in tqdm(enumerate(df_insertions.iterrows()), desc='Probe: ', leave=False):
            probe = ins['probe']
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            spikes = alfio.load_object(spike_sorting_path, 'spikes')
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = dut.remap_region(clusters.atlas_id, br=brainreg)
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            across_probes['regions'].extend(beryl_reg)
            across_probes['clusters'].extend(spikes.clusters if i_probe == 0 else
                                             (spikes.clusters + max(across_probes['clusters']) + 1))
            across_probes['times'].extend(spikes.times)
            across_probes['qc_pass'].extend(qc_pass)
        across_probes = {k: np.array(v) for k, v in across_probes.items()}
        # warnings.filterwarnings('ignore')
        if kwargs['single_region']:
            regions = [[k] for k in np.unique(across_probes['regions'])]
        else:
            regions = [np.unique(across_probes['regions'])]
        df_insertions_iterrows = pd.DataFrame.from_dict(
            {'1': 'mergedProbes'}, orient='index', columns=['probe']).iterrows()
    elif wideFieldImaging_dict is None:
        df_insertions_iterrows = df_insertions.iterrows()
    else:
        regions = wideFieldImaging_dict['atlas'].acronym.values
        df_insertions_iterrows = pd.DataFrame.from_dict(
            {'1': 'mergedProbes'}, orient='index', columns=['probe']).iterrows()

    # ---------------------------------------------------------------------------------------------
    # loop through probes
    # ---------------------------------------------------------------------------------------------
    for i, ins in tqdm(df_insertions_iterrows, desc='Probe: ', leave=False):
        probe = ins['probe']
        if not kwargs['merged_probes']:
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            spikes = alfio.load_object(spike_sorting_path, 'spikes')
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = dut.remap_region(clusters.atlas_id, br=brainreg)
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            regions = np.unique(beryl_reg)

        # -----------------------------------------------------------------------------------------
        # loop through regions
        # -----------------------------------------------------------------------------------------
        for region in tqdm(regions, desc='Region: ', leave=False):

            if region == 'root':
                continue

            # select good units from this insertion/region
            if kwargs['merged_probes'] and wideFieldImaging_dict is None:
                reg_mask = np.isin(across_probes['regions'], region)
                reg_clu_ids = np.argwhere(reg_mask & across_probes['qc_pass']).flatten()
            elif wideFieldImaging_dict is None:
                reg_mask = beryl_reg == region
                reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            else:
                region_labels = []
                reg_lab = wideFieldImaging_dict['atlas'][wideFieldImaging_dict['atlas'].acronym == region].label.values.squeeze()
                if 'left' in kwargs['wfi_hemispheres']:
                    region_labels.append(reg_lab)
                if 'right' in kwargs['wfi_hemispheres']:
                    region_labels.append(-reg_lab)
                reg_mask = np.isin(wideFieldImaging_dict['regions'], region_labels)
                reg_clu_ids = np.argwhere(reg_mask)
            N_units = len(reg_clu_ids)

            # skip region if not enough good units
            if N_units < kwargs['min_units']:
                continue

            # format spike data for regression models
            if kwargs['merged_probes'] and wideFieldImaging_dict is None:
                spikemask = np.isin(across_probes['clusters'], reg_clu_ids)
                regspikes = across_probes['times'][spikemask]
                regclu = across_probes['clusters'][spikemask]
                arg_sortedSpikeTimes = np.argsort(regspikes)
                spike_times_list, spikes_list = dutc.get_spike_data_per_trial(
                    regspikes[arg_sortedSpikeTimes], regclu[arg_sortedSpikeTimes],
                    interval_beg_times - kwargs['n_bins_lag'] * kwargs['binsize'],  # incl window
                    interval_end_times, kwargs['binsize'])
            elif wideFieldImaging_dict is None:
                spikemask = np.isin(spikes.clusters, reg_clu_ids)
                regspikes = spikes.times[spikemask]
                regclu = spikes.clusters[spikemask]
                spike_times_list, spikes_list = dutc.get_spike_data_per_trial(
                    regspikes, regclu,
                    interval_beg_times - kwargs['n_bins_lag'] * kwargs['binsize'],  # incl window
                    interval_end_times, kwargs['binsize'])
            else:
                raise NotImplementedError
                # frames_idx = wideFieldImaging_dict['timings'][kwargs['align_time']].values
                # frames_idx = np.sort(
                #     frames_idx[:, None] + np.arange(0, kwargs['wfi_nb_frames'], np.sign(kwargs['wfi_nb_frames'])),
                #     axis=1,
                # )
                # binned = np.take(wideFieldImaging_dict['activity'][:, reg_mask], frames_idx, axis=0)
                # binned = binned.reshape(binned.shape[0], -1).T

            # -------------------------------------------------------------------------------------
            # loop through real/pseudo/imposter sessions
            # -------------------------------------------------------------------------------------
            for pseudo_id in pseudo_ids:

                save_path = dutc.get_save_path(
                    pseudo_id, subject, eid, probe,
                    str(np.squeeze(region)) if kwargs['single_region'] else 'allRegions',
                    output_path=kwargs['output_path'],
                    time_window=kwargs['time_window'],
                    today=kwargs['today'],
                    target=kwargs['target'],
                    add_to_saving_path=kwargs['add_to_saving_path']
                )
                if os.path.exists(save_path):
                    logging.log(
                        logging.DEBUG,
                        '%s (id=%i) already exists, skipping' % (region, pseudo_id))
                    continue

                # create pseudo session/imposter session targets when necessary
                if pseudo_id > 0:
                    if kwargs['use_imposter_session']:
                        raise NotImplementedError
                        # pseudosess = dut.generate_imposter_session(
                        #     kwargs['imposterdf'], eid, trialsdf.index.size, nbSampledSess=10)
                        # pseudomask = mask & (pseudosess.choice != 0)
                    else:
                        raise NotImplementedError(
                            "Must use imposter session for decoding %s" % kwargs['target'])

                    # msub_pseudo_tvec = dut.compute_target(
                    #     kwargs['target'], subject, subjeids, eid,
                    #     kwargs['modelfit_path'],
                    #     binarization_value=kwargs['binarization_value'],
                    #     modeltype=kwargs['model'],
                    #     beh_data_test=pseudosess, one=one,
                    #     behavior_data_train=behavior_data_train
                    # )[pseudomask]

                else:
                    target_list = target_val_list

                # fit decoder on multiple runs
                fit_results = []
                for i_run in range(kwargs['n_runs']):

                    # build predictor matrix
                    predictor_list = [
                        dutc.build_predictor_matrix(s.T, kwargs['n_bins_lag']) for s in spikes_list
                    ]

                    # train models
                    fit_result = dutc.decode(
                        predictor_list, target_list,
                        decoder=kwargs['estimator'],
                        hyperparameter_grid=kwargs['hyperparameter_grid'],
                        shuffle=kwargs['shuffle'],
                        nFolds=5,
                        outer_cv=True,
                        save_predictions=True if pseudo_id == -1 else False,
                        rng_seed=i_run,
                    )

                    # fit_result['mask'] = mask & (trialsdf.choice != 0)
                    fit_result['df'] = trialsdf if pseudo_id == -1 else pseudosess
                    fit_result['pseudo_id'] = pseudo_id
                    fit_result['run_id'] = i_run
                    fit_results.append(fit_result)

                filenames.append(dutc.save_region_results(
                    fit_results, pseudo_id, subject, eid, probe,
                    str(np.squeeze(region)) if kwargs['single_region'] else 'allRegions',
                    N_units, save_path=save_path))

    return filenames
