import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ibllib.atlas import BrainRegions

from braindelphi.decoding.functions.process_inputs import build_predictor_matrix
from braindelphi.decoding.functions.process_inputs import select_ephys_regions
from braindelphi.decoding.functions.process_inputs import select_widefield_imaging_regions
from braindelphi.decoding.functions.process_inputs import preprocess_ephys
from braindelphi.decoding.functions.process_inputs import proprocess_widefield_imaging
from braindelphi.decoding.functions.process_targets import compute_beh_target
from braindelphi.decoding.functions.process_targets import get_target_data_per_trial_error_check
from braindelphi.decoding.functions.decoder import decode_cv
from braindelphi.decoding.functions.utils import compute_mask
from braindelphi.decoding.functions.utils import save_region_results


def fit_eid(neural_dict, trials_df, metadata, dlc_dict=None, pseudo_ids=[-1], **kwargs):
    """
    Parameters
    ----------
    neural_dict : dict
        keys: 'trials_df', 'spk_times', 'spk_clu', 'clu_regions', 'clu_qc', 'clu_df'
    dlc_dict: dlc_dict with dlc, pupil dilation, wheel velocity
        'times', 'values'
    metadata
        'eid', 'eid_train', 'subject', 'probe'
    eid: eid of session
    bwmdf: dataframe of bwm session
    pseudo_id: whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered.
    can not be 0
    nb_runs: nb of independent runs performed. this was added after consequent variability was observed across runs.
    modelfit_path: outputs of behavioral fits
    output_path: outputs of decoding fits
    one: ONE object -- this is not to be used with dask, this option is given for debugging purposes
    sessiondf: the behavioral and neural dataframe when you want to bypass the bwm encoding phase
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
        min_units : int
        qc_criteria : float
        single_region : bool
        merged_probes : bool
        criterion : str
        min_behav_trials : int
        min_rt : float
        no_unbias : bool
        neural_dtype : str
        today : str
        output_path : str
            outputs of decoding fits
        add_to_saving_path : str
        imposter_df : pandas.DataFrame

    """

    print(f'Working on eid : %s' % metadata['eid'])
    filenames = []  # this will contain paths to saved decoding results for this eid

    if 0 in pseudo_ids:
        raise ValueError(
            'pseudo id can be -1 (actual session) or strictly greater than 0 (pseudo session)')

    # TODO: stim, choice, feedback, etc
    if kwargs['target'] == 'prior':
        target_vals_list = compute_beh_target(trials_df, metadata, **kwargs)
        mask_target = np.ones(len(target_vals_list), dtype=bool)
    else:
        _, target_vals_list, mask_target = get_target_data_per_trial_error_check(
            dlc_dict['times'], dlc_dict['values'], trials_df, kwargs['align_event'],
            kwargs['align_interval'], kwargs['binsize'])

    mask = compute_mask(trials_df, **kwargs) & mask_target

    n_trials = np.sum(mask)
    if n_trials <= kwargs['min_behav_trials']:
        msg = 'session contains %i trials, below the threshold of %i' % (
            n_trials, kwargs['min_behav_trials'])
        logging.exception(msg)
        return filenames

    # select brain regions from beryl atlas to loop over
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(regressors['clu_regions'], mapping='Beryl')
    regions = (
        [[k] for k in np.unique(beryl_reg)]
        if kwargs['single_region']
        else [np.unique(beryl_reg)]
    )

    for region in tqdm(regions, desc='Region: ', leave=False):

        if kwargs['neural_dtype'] == 'ephys':
            reg_clu_ids = select_ephys_regions(neural_dict, beryl_reg, region, **kwargs)
        elif kwargs['neural_dtype'] == 'widefield':
            reg_clu_ids = select_widefield_imaging_regions()
        else:
            raise NotImplementedError

        N_units = len(reg_clu_ids)
        if N_units < kwargs['min_units']:
            continue

        if kwargs['neural_dtype'] == 'ephys':
            msub_binned = preprocess_ephys(reg_clu_ids, neural_dict, trials_df, **kwargs)
        elif kwargs['neural_dtype'] == 'widefield':
            msub_binned = proprocess_widefield_imaging()
        else:
            raise NotImplementedError

        if kwargs['simulate_neural_data']:
            raise NotImplementedError

        for pseudo_id in pseudo_ids:

            # create pseudo session when necessary
            if pseudo_id > 0:
                # todo  generate pseudo session
                pseudo_targets = dut.compute_target(
                    kwargs['target'],
                    subject,
                    subjeids,
                    eid,
                    kwargs['modelfit_path'],
                    binarization_value=kwargs['binarization_value'],
                    modeltype=kwargs['model'],
                    beh_data_test=pseudosess,
                    one=one,
                    behavior_data_train=behavior_data_train)[pseudomask]

            if kwargs['compute_neurometric']:  # compute prior for neurometric curve
                raise NotImplementedError

            # make design matrix if multiple bins per trial
            bins_per_trial = len(msub_binned[0])
            if bins_per_trial == 1:
                Xs = msub_binned
            else:
                Xs = [build_predictor_matrix(s.T, kwargs['n_bins_lag']) for s in msub_binned]

            # compute
            fit_results = []
            for i_run in range(kwargs['nb_runs']):
                fit_result = decode_cv(
                    ys=target_vals_list if (pseudo_id == -1) else pseudo_targets,
                    Xs=Xs,
                    estimator=kwargs['estimator'],
                    use_openturns=kwargs['use_openturns'],
                    target_distribution=target_distribution,
                    bin_size_kde=kwargs['bin_size_kde'],
                    balanced_continuous_target=kwargs['balanced_continuous_target'],
                    estimator_kwargs=kwargs['estimator_kwargs'],
                    hyperparam_grid=kwargs['hyperparam_grid'],
                    save_binned=kwargs['save_binned'],
                    shuffle=kwargs['shuffle'],
                    balanced_weight=kwargs['balanced_weight'],
                    normalize_input=kwargs['normalize_input'],
                    normalize_output=kwargs['normalize_output'])
                fit_result['mask'] = mask & (trialsdf.choice != 0)
                fit_result['df'] = trialsdf if pseudo_id == -1 else pseudosess
                fit_result['pseudo_id'] = pseudo_id
                fit_result['run_id'] = i_run

                # compute neurometric curves
                if kwargs['compute_neurometric']:
                    raise NotImplementedError
                    # fit_result['full_neurometric'], fit_result['fold_neurometric'] = \
                    #     get_neurometric_parameters(
                    #         fit_result,
                    #         trialsdf=trialsdf_neurometric,
                    #         one=one,
                    #         compute_on_each_fold=kwargs['compute_on_each_fold'],
                    #         force_positive_neuro_slopes=kwargs['compute_on_each_fold'])
                else:
                    fit_result['full_neurometric'] = None
                    fit_result['fold_neurometric'] = None
                fit_results.append(fit_result)

            filenames.append(
                save_region_results(fit_results,
                                    pseudo_id,
                                    region,
                                    N_units,
                                    metadata,
                                    **kwargs)
            )
                    #output_path=kwargs['output_path'],
                    #time_window=kwargs['time_window'],
                    #today=kwargs['today'],
                    #target=kwargs['target'],
                    #add_to_saving_path=kwargs['add_to_saving_path']))

    return filenames


if __name__ == '__main__':
    file = 'example_neural_and_behavioral_data.pkl'
    import pickle
    regressors = pickle.load(open(file, 'rb'))
    trials_df = regressors['trialsdf']
    neural_dict = regressors
    metadata = {'eids_train':['test'], 'eid': 'test', 'subject':'mouse_name'}

