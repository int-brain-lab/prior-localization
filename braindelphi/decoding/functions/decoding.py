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


def fit_eid(neural_dict, trials_df, metadata, dlc_dict=None, pseudo_ids=[-1], **kwargs):
    """
    Parameters
    ----------
    dlc_dict: dlc_dict with dlc, pupil dilation, wheel velocity
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
        wide_field_imaging : bool
        today : str
        output_path : str
            outputs of decoding fits
        add_to_saving_path : str
        imposter_df : pandas.DataFrame
    """

    if 0 in pseudo_ids:
        raise ValueError(
            'pseudo id can be -1 (actual session) or strictly greater than 0 (pseudo session)')

    # todo move compute_target to process_targets
    # todo make compute_mask from trials_df and 'others' and put this in utils

    from process_targets import compute_beh_target
    tvec = compute_beh_target(trials_df)

    from ibllib.atlas import BrainRegions
    brainreg = BrainRegions()

    from utils import compute_mask
    mask_trials_df = compute_mask(trials_df)
    # todo multiply the two masks

    filenames = []
    if len(tvec[mask]) <= kwargs['min_behav_trials']:
        return filenames

    print(f'Working on eid : %s' % metadata['eid'])

    # warnings.filterwarnings('ignore')
    regions = (
        [[k] for k in np.unique(regressors['clu_regions'])]
        if kwargs['single_region']
        else [np.unique(regressors['clu_regions'])]
    )

    probe = metadata['probe']
    beryl_reg = brainreg.acronym2acronym(regressors['clu_regions'], mapping='Beryl')
    qc_pass = (regressors['clu_qc'] >= kwargs['qc_criteria']).values
    regions = np.unique(beryl_reg)

    for region in tqdm(regions, desc='Region: ', leave=False):

        from process_inputs import select_ephys_regions
        from process_inputs import select_widefield_imaging_regions
        reg_clu_ids = (
            select_widefield_imaging_regions() if kwargs['wide_field_imaging']
            else select_ephys_regions()
        )

        N_units = len(reg_clu_ids)
        if N_units < kwargs['min_units']:
            continue
        intervals = np.vstack([
            trialsdf[kwargs['align_time']] + kwargs['time_window'][0],
            trialsdf[kwargs['align_time']] + kwargs['time_window'][1]
        ]).T

        from process_inputs import preprocess_ephys
        from process_inputs import proprocess_widefield_imaging
        msub_binned = (
            proprocess_widefield_imaging() if kwargs['wide_field_imaging']
            else preprocess_ephys()
        ).T

        if kwargs['simulate_neural_data']:
            raise NotImplementedError

        for pseudo_id in pseudo_ids:
            if pseudo_id > 0:  # create pseudo session when necessary
                msub_pseudo_tvec = dut.compute_target(
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
                # do neurometric stuff

            fit_results = []
            for i_run in range(kwargs['nb_runs']):
                fit_result = dut.regress_target(
                    (tvec[mask & (nb_trialsdf.choice != 0)] if
                     (pseudo_id == -1) else msub_pseudo_tvec),
                    (msub_binned[nb_trialsdf.choice != 0] if
                     (pseudo_id == -1) else msub_binned[pseudosess[mask].choice != 0]),
                    kwargs['estimator'],
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

            filenames.append(
                save_region_results(
                    fit_results,
                    pseudo_id,
                    subject,
                    eid,
                    probe,
                    str(np.squeeze(region)) if kwargs['single_region'] else 'allRegions',
                    N_units,
                    output_path=kwargs['output_path'],
                    time_window=kwargs['time_window'],
                    today=kwargs['today'],
                    target=kwargs['target'],
                    add_to_saving_path=kwargs['add_to_saving_path']))

    return filenames


if __name__ == '__main__':
    file = '/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/decoding/cache/ibl_witten_32/7502ae93-7437-4bcd-9e14-d73b51193656/2022-05-22_regressors.pkl'
    import pickle
    regressors = pickle.load(open(file, 'rb'))
