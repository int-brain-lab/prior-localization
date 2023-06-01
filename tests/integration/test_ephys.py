import pytest
import pandas as pd
import sys
from tests.integration.ephys_settings import kwargs
from prior_pipelines.decoding.functions.decoding import fit_eid
import numpy as np
from prior_pipelines.pipelines.utils_ephys import load_ephys
from one.api import ONE
from brainwidemap import bwm_query
import os

one = ONE(mode='local')
bwm_df = bwm_query().set_index(["subject", "eid"])

def test_of_test():   
    return True

# decoding function
def decode(regressors, metadata, nb_pseudo_sessions=3):
    if kwargs['neural_dtype'] == 'widefield':
        trials_df, neural_dict = regressors
    else:
        trials_df, neural_dict = regressors['trials_df'], regressors

    # pseudo_id=-1 decodes the prior of the session, pseudo_id>0 decodes pseudo-priors
    pseudo_ids = np.concatenate((-np.ones(1), np.arange(1, nb_pseudo_sessions))).astype('int64')
    results_fit_eid = fit_eid(neural_dict=neural_dict, trials_df=trials_df, metadata=metadata,
                            pseudo_ids=pseudo_ids, **kwargs)
    return results_fit_eid # these are filenames


@pytest.mark.parametrize("MERGED_PROBES", [True, False])
def test_output_of_ephys_decoding(MERGED_PROBES):

    eid = "56956777-dca5-468c-87cb-78150432cc57"

    # select session, eid, and pids and subject of interest
    session_df = bwm_df.xs(eid, level="eid")
    subject = session_df.index[0]
    pids = session_df.pid.to_list()
    probe_names = session_df.probe_name.to_list()
    metadata = {
        "subject": subject,
        "eid": eid,
        "probe_name": 'merged_probes' if MERGED_PROBES else None,
        "ret_qc": 1,
    }

    # launch decoding
    results_fit_eid = []
    if MERGED_PROBES:
        regressors = load_ephys(eid, pids, one=one, pnames=probe_names, **{'ret_qc':True})
        results_fit_eid.extend(decode(regressors, metadata))
    else:
        probe_names = session_df.probe_name.to_list()
        for (pid, probe_name) in zip(pids, probe_names):
            regressors =  load_ephys(eid, [pid], one=one, pnames=[probe_name], ret_qc=1)
            metadata['probe_name'] = probe_name
            results_fit_eid.extend(decode(regressors, metadata))

    import pickle
    for f in results_fit_eid:        
        predicteds = pickle.load(open(f, 'rb'))
        targets = pickle.load(open(f.parent.joinpath(f.name.replace('_to_be_tested', '')), 'rb'))
        for (fit_pred, fit_targ) in zip(predicteds['fit'], targets['fit']):
            assert (fit_pred['Rsquared_test_full'] == fit_targ['Rsquared_test_full'])
            assert (fit_pred['predictions_test'] == fit_targ['predictions_test'])
        f.unlink() # remove predicted path

    print('job successful')