from pathlib import Path
import numpy as np

from behavior_models.utils import build_path


def create_neural_path(output_path, date, neural_dtype, subject, session_id, probe,
                       region_str, target, time_window,  pseudo_ids, add_to_path=None):
    full_path = Path(output_path).joinpath('neural', date, neural_dtype, subject, session_id, probe)
    full_path.mkdir(exist_ok=True, parents=True)

    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if isinstance(pseudo_ids, np.ndarray) else str(pseudo_id)
    time_str = f'{time_window[0]}_{time_window[1]}'.replace('.', '_')
    file_name = f'{region_str}_target_{target}_timeWindow_{time_str}_pseudo_id_{pseudo_str}'
    if add_to_path:
        for k, v in add_to_path.items():
            file_name = f'{file_name}_{k}_{v}'
    full_path.joinpath(f'{file_name}.pkl')
    return full_path


def check_bhv_fit_exists(subject, model, eids, resultpath, modeldispatcher, single_zeta):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    path_results_mouse = 'model_%s_' % modeldispatcher[model] + 'single_zeta_' * single_zeta
    trunc_eids = [eid.split('-')[0] for eid in eids]
    filen = build_path(path_results_mouse, trunc_eids)
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    fullpath = subjmodpath.joinpath(filen)
    return fullpath.exists(), fullpath

