import numpy as np
import pandas as pd
import functions.utils as dut
import brainbox.io.one as bbone
import models.utils as mut
from pathlib import Path
from functions.utils import save_region_results
from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_session
import one.alf.io as alfio
from functions.neurometric import get_neurometric_parameters
from tqdm import tqdm
import pickle

def fit_eid(eid, bwmdf, pseudo_ids=[-1], sessiondf=None, wideFieldImaging_dict=None, **kwargs):
    """
    Parameters
    ----------
    eid: eid of session
    bwmdf: dataframe of bwm session
    pseudo_id: whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered.
    can not be 0
    nb_runs: nb of independent runs performed. this was added after consequent variability was observed across runs.
    modelfit_path: outputs of behavioral fits
    output_path: outputs of decoding fits
    one: ONE object -- this is not to be used with dask, this option is given for debugging purposes
    sessiondf: the behavioral and neural dataframe when you want to bypass the bwm encoding phase
    """

    if ((wideFieldImaging_dict is None and kwargs['wide_field_imaging']) or
            (wideFieldImaging_dict is not None and not kwargs['wide_field_imaging'])):
        raise ValueError('wideFieldImaging_dict must be defined for wide_field_imaging and reciprocally')

    if kwargs['wide_field_imaging'] and kwargs['wfi_nb_frames'] == 0:
        raise ValueError('wfi_nb_frames can not be 0. it is a signed non-null integer')

    if 0 in pseudo_ids:
        raise ValueError('pseudo id can be -1 (actual session) or strictly greater than 0 (pseudo session)')

    one = ONE() if kwargs['one'] is None else kwargs['one']
    if sessiondf is None:
        df_insertions = bwmdf.loc[bwmdf['eid'] == eid]
        subject = df_insertions['subject'].to_numpy()[0]
        if kwargs['beh_mouseLevel_training']:
            subjeids = bwmdf.loc[bwmdf['subject'] == subject]['eid'].unique()
        else:
            subjeids = np.array([eid])
        brainreg = dut.BrainRegions()
        behavior_data = mut.load_session(eid, one=one)
        behavior_data_train = None
    else:
        subject = sessiondf.subject.unique()[0]
        subjeids = sessiondf.eid.unique()
        behavior_data_train = sessiondf
        behavior_data = {k: np.array(v) for (k, v) in sessiondf[sessiondf.session_to_decode].to_dict('list').items()}

    try:
        tvec = dut.compute_target(kwargs['target'], subject, subjeids, eid, kwargs['modelfit_path'],
                                  binarization_value=kwargs['binarization_value'],
                                  modeltype=kwargs['model'], behavior_data_train=behavior_data_train,
                                  beh_data_test=behavior_data, one=one)
        '''
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(tvec, label='pLeft')
        plt.plot((behavior_data['choice'] == 1) * 1 + (behavior_data['choice'] == 0) * 0.5, label='action')
        plt.legend()
        plt.draw()
        plt.show()
        '''
    except ValueError:
        print('Model not fit.')
        tvec = dut.compute_target(kwargs['target'], subject, subjeids, eid, kwargs['modelfit_path'],
                                  binarization_value=kwargs['binarization_value'],
                                  modeltype=kwargs['model'], one=kwargs['one'], beh_data_test=behavior_data)

    try:
        if sessiondf is None:
            trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])
        else:
            trialsdf = sessiondf[sessiondf.session_to_decode]
            if 'react_times' in trialsdf.columns:
                trialsdf = trialsdf.drop('react_times', axis=1)
        if len(trialsdf) != len(tvec):
            raise IndexError
    except IndexError:
        raise IndexError('Problem in the dimensions of dataframe of session')
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf[kwargs['align_time']]
    mask = trialsdf[kwargs['align_time']].notna() & trialsdf['firstMovement_times'].notna()
    if kwargs['no_unbias']:
        mask = mask & (trialsdf.probabilityLeft != 0.5).values
    if kwargs['min_rt'] is not None:
        mask = mask & (~(trialsdf.react_times < kwargs['min_rt'])).values
    nb_trialsdf = trialsdf[mask]  # take out when mouse doesn't perform any action

    if kwargs['balanced_weight'] and kwargs['balanced_continuous_target']:
        if (kwargs['no_unbias'] and not kwargs['use_imposter_session_for_balancing']
                and (kwargs['model'] == dut.optimal_Bayesian)):
            with open(kwargs['decoding_path'].joinpath('targetpLeft_optBay_%s.pkl' %
                                                       str(kwargs['bin_size_kde']).replace('.', '_')), 'rb') as f:
                target_distribution = pickle.load(f)
        elif not kwargs['use_imposter_session_for_balancing'] and (kwargs['model'] == dut.optimal_Bayesian):
            target_distribution, _ = dut.get_target_pLeft(nb_trials=trialsdf.index.size, nb_sessions=250,
                                                          take_out_unbiased=False, bin_size_kde=kwargs['bin_size_kde'])
        else:
            subjModel = {'modeltype': kwargs['model'], 'subjeids': subjeids, 'subject': subject,
                         'modelfit_path': kwargs['modelfit_path'], 'imposterdf': kwargs['imposterdf'],
                         'use_imposter_session_for_balancing': kwargs['use_imposter_session_for_balancing'],
                         'eid': eid, 'target': kwargs['target']}
            target_distribution, allV_t = dut.get_target_pLeft(nb_trials=trialsdf.index.size, nb_sessions=250,
                                                               take_out_unbiased=kwargs['no_unbias'],
                                                               bin_size_kde=kwargs['bin_size_kde'], subjModel=subjModel)
    else:
        target_distribution = None

    filenames = []
    if len(tvec[mask]) <= kwargs['min_behav_trials']:
        return filenames

    print(f'Working on eid : {eid}')

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
        df_insertions_iterrows = pd.DataFrame.from_dict({'1': 'mergedProbes'},
                                                        orient='index',
                                                        columns=['probe']).iterrows()
    elif wideFieldImaging_dict is None:
        df_insertions_iterrows = df_insertions.iterrows()
    else:
        regions = wideFieldImaging_dict['atlas'].acronym.values
        df_insertions_iterrows = pd.DataFrame.from_dict({'1': 'mergedProbes'},
                                                        orient='index',
                                                        columns=['probe']).iterrows()

    for i, ins in tqdm(df_insertions_iterrows, desc='Probe: ', leave=False):
        probe = ins['probe']
        if not kwargs['merged_probes']:
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            spikes = alfio.load_object(spike_sorting_path, 'spikes')
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = dut.remap_region(clusters.atlas_id, br=brainreg)
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        for region in tqdm(regions, desc='Region: ', leave=False):
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
            if N_units < kwargs['min_units']:
                continue
            # or get_spike_count_in_bins
            if np.any(np.isnan(nb_trialsdf[kwargs['align_time']])):
                # if this happens, verify scrub of NaN values in all aign times before get_spike_counts_in_bins
                raise ValueError('this should not happen')
            intervals = np.vstack([nb_trialsdf[kwargs['align_time']] + kwargs['time_window'][0],
                                   nb_trialsdf[kwargs['align_time']] + kwargs['time_window'][1]]).T

            if kwargs['merged_probes'] and wideFieldImaging_dict is None:
                spikemask = np.isin(across_probes['clusters'], reg_clu_ids)
                regspikes = across_probes['times'][spikemask]
                regclu = across_probes['clusters'][spikemask]
                arg_sortedSpikeTimes = np.argsort(regspikes)
                binned, _ = get_spike_counts_in_bins(regspikes[arg_sortedSpikeTimes],
                                                     regclu[arg_sortedSpikeTimes],
                                                     intervals)
            elif wideFieldImaging_dict is None:
                spikemask = np.isin(spikes.clusters, reg_clu_ids)
                regspikes = spikes.times[spikemask]
                regclu = spikes.clusters[spikemask]
                binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
            else:
                frames_idx = wideFieldImaging_dict['timings'][kwargs['align_time']].values
                frames_idx = np.sort(
                    frames_idx[:, None] + np.arange(0, kwargs['wfi_nb_frames'], np.sign(kwargs['wfi_nb_frames'])),
                    axis=1,
                )
                binned = np.take(wideFieldImaging_dict['activity'][:, reg_mask], frames_idx, axis=0)
                binned = binned.reshape(binned.shape[0], -1).T

            msub_binned = binned.T  # number of trials x nb bins

            if kwargs['simulate_neural_data']:
                x_range = np.arange(len(msub_binned))
                polyfit = np.polyfit(x_range, msub_binned, deg=3)
                ypred = np.polyval(polyfit, x_range[:, None])
                ypred = np.maximum(
                    np.nan_to_num((tvec[mask][:, None] * ypred) / (tvec[mask][:, None] * ypred).mean(axis=0)
                                  * np.mean(msub_binned, axis=0)[None], 0), 0
                )
                msub_binned = np.random.poisson(ypred)
                '''
                from matplotlib import pyplot as plt
                plt.figure()
                iplot=12
                plt.plot(msub_binned[:, iplot]); plt.plot(ypred[:, iplot])
                plt.draw(); plt.show()
                '''

            if len(msub_binned.shape) > 2:
                raise ValueError('Multiple bins are being calculated per trial,'
                                 'may be due to floating point representation error.'
                                 'Check window.')

            for pseudo_id in pseudo_ids:
                if pseudo_id > 0:  # create pseudo session when necessary
                    if kwargs['use_imposter_session']:
                        if not kwargs['constrain_imposter_session_with_beh']:
                            pseudosess = dut.generate_imposter_session(kwargs['imposterdf'], eid,
                                                                       trialsdf.index.size, nbSampledSess=10)
                        else:
                            feedback_0contrast = trialsdf[(trialsdf.contrastLeft == 0).values +
                                                          (trialsdf.contrastRight == 0).values].feedbackType.mean()

                            pseudosess_s = [dut.generate_imposter_session(kwargs['imposterdf'], eid,
                                                                       trialsdf.index.size, nbSampledSess=10)
                                            for _ in range(50)]
                            feedback_pseudo_0cont = [pseudo[(pseudo.contrastLeft == 0).values +
                                                            (pseudo.contrastRight == 0).values].feedbackType.mean()
                                                     for pseudo in pseudosess_s]
                            pseudosess = pseudosess_s[np.argmin(np.abs(np.array(feedback_pseudo_0cont)
                                                                       - feedback_0contrast))]
                        pseudomask = mask & (pseudosess.choice != 0)
                    else:
                        pseudosess = generate_pseudo_session(trialsdf, generate_choices=False)
                        if kwargs['model'].name == 'actKernel':
                            subjModel = {'modeltype': kwargs['model'], 'subjeids': subjeids, 'subject': subject,
                                         'modelfit_path': kwargs['modelfit_path'], 'eid': eid}
                            pseudosess['choice'] = dut.generate_choices(pseudosess, trialsdf, subjModel)
                        else:
                            pseudosess['choice'] = trialsdf.choice
                        pseudomask = mask & (trialsdf.choice != 0)

                    msub_pseudo_tvec = dut.compute_target(kwargs['target'], subject, subjeids, eid,
                                                          kwargs['modelfit_path'],
                                                          binarization_value=kwargs['binarization_value'],
                                                          modeltype=kwargs['model'],
                                                          beh_data_test=pseudosess, one=one,
                                                          behavior_data_train=behavior_data_train)[pseudomask]

                if kwargs['compute_neurometric']:  # compute prior for neurometric curve
                    trialsdf_neurometric = nb_trialsdf.reset_index() if (pseudo_id == -1) else \
                        pseudosess[pseudomask].reset_index()
                    if kwargs['model'] is not None:
                        blockprob_neurometric = dut.compute_target('pLeft', subject, subjeids, eid,
                                                                   kwargs['modelfit_path'],
                                                                   binarization_value=kwargs['binarization_value'],
                                                                   modeltype=kwargs['model'],
                                                                   beh_data_test=trialsdf if pseudo_id == -1 else pseudosess,
                                                                   behavior_data_train=behavior_data_train, one=one)

                        trialsdf_neurometric['blockprob_neurometric'] = np.stack(
                            [np.greater_equal(
                                blockprob_neurometric[(mask & (trialsdf.choice != 0) if pseudo_id == -1 else pseudomask)],
                                border).astype(int)
                             for border in kwargs['border_quantiles_neurometric']
                             ]).sum(0)

                    else:
                        blockprob_neurometric = trialsdf_neurometric['probabilityLeft'].replace(0.2, 0).replace(0.8, 1)
                        trialsdf_neurometric['blockprob_neurometric'] = blockprob_neurometric

                fit_results = []
                for i_run in range(kwargs['nb_runs']):
                    fit_result = dut.regress_target((tvec[mask & (nb_trialsdf.choice != 0)]
                                                     if (pseudo_id == -1) else msub_pseudo_tvec),
                                                    (msub_binned[nb_trialsdf.choice != 0]
                                                     if (pseudo_id == -1) else msub_binned[pseudosess[mask].choice != 0]),
                                                    kwargs['estimator'],
                                                    use_openturns=kwargs['use_openturns'],
                                                    target_distribution=target_distribution,
                                                    bin_size_kde=kwargs['bin_size_kde'],
                                                    balanced_continuous_target=kwargs['balanced_continuous_target'],
                                                    estimator_kwargs=kwargs['estimator_kwargs'],
                                                    hyperparam_grid=kwargs['hyperparam_grid'],
                                                    save_binned=kwargs['save_binned'], shuffle=kwargs['shuffle'],
                                                    balanced_weight=kwargs['balanced_weight'],
                                                    normalize_input=kwargs['normalize_input'],
                                                    normalize_output=kwargs['normalize_output'])
                    fit_result['mask'] = mask & (trialsdf.choice != 0)
                    fit_result['df'] = trialsdf if pseudo_id == -1 else pseudosess
                    fit_result['pseudo_id'] = pseudo_id
                    fit_result['run_id'] = i_run
                    # neurometric curve
                    if kwargs['compute_neurometric']:
                        fit_result['full_neurometric'], fit_result['fold_neurometric'] = \
                            get_neurometric_parameters(fit_result,
                                                       trialsdf=trialsdf_neurometric,
                                                       one=one,
                                                       compute_on_each_fold=kwargs['compute_on_each_fold'],
                                                       force_positive_neuro_slopes=kwargs['compute_on_each_fold'])
                    else:
                        fit_result['full_neurometric'] = None
                        fit_result['fold_neurometric'] = None
                    fit_results.append(fit_result)

                filenames.append(save_region_results(fit_results, pseudo_id, subject, eid, probe,
                                                     str(np.squeeze(region)) if kwargs['single_region'] else 'allRegions',
                                                     N_units, output_path=kwargs['output_path'],
                                                     time_window=kwargs['time_window'],
                                                     today=kwargs['today'],
                                                     target=kwargs['target'],
                                                     add_to_saving_path=kwargs['add_to_saving_path']))

    return filenames