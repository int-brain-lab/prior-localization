import sys
import os

MODELS_PATH = '/home/users/bensonb/international-brain-lab/behavior_models'
if not MODELS_PATH in sys.path:                                                                              
     sys.path.insert(0, MODELS_PATH)
DECODING_PATH = '/home/users/bensonb/international-brain-lab/prior-localization/decoding'
if not DECODING_PATH in sys.path:                                                                              
     sys.path.insert(0, DECODING_PATH)
GROUP_HOME = os.environ['GROUP_HOME']

import pickle
import logging
import numpy as np
import pandas as pd
import decoding_utils as dut
import brainbox.io.one as bbone
import sklearn.linear_model as sklm
import models.utils as mut
from pathlib import Path
from datetime import date
from one.api import ONE
from one.params import CACHE_DIR_DEFAULT
from models.expSmoothing_prevAction import expSmoothing_prevAction
from models.optimalBayesian import optimal_Bayesian
# from brainbox.singlecell import calculate_peths
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_session
from brainbox.metrics.single_units import quick_unit_metrics
from decoding_stimulus_neurometric_fit import get_neurometric_parameters

from tqdm import tqdm
from ibllib.atlas import AllenAtlas

logger = logging.getLogger('ibllib')
logger.disabled = True

strlut = {sklm.Lasso: 'Lasso',
          sklm.LassoCV: 'LassoCV',
          sklm.Ridge: 'Ridge',
          sklm.RidgeCV: 'RidgeCV',
          sklm.LinearRegression: 'PureLinear',
          sklm.LogisticRegression: 'Logistic'}

# %% Run param definitions

# aligned -> histology was performed by one experimenter
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'aligned-behavior'  # aligned and behavior
MODEL = None  # None or expSmoothing_prevAction or dut.modeldispatcher
DATE = str(date.today())
MODELFIT_PATH = os.path.join(GROUP_HOME,'bensonb/international-brain-lab/prior-localization/behavior/')
OUTPUT_PATH = os.path.join(GROUP_HOME,'bensonb/international-brain-lab/prior-localization/decoding/')

TARGET = 'pLeft'  # 'pLeft','prior','choice','feedback','signcont'
CONTROL_FEATURES = [] # subset of the following including empty: 'pLeft','choice','feedback','signcont'
ALIGN_TIME = 'goCue_times'# 'feedback_times'
TIME_WINDOW = (-0.6, -0.2)  # (-0.6, -0.2), (0, 0.1)
MIN_UNITS = 10
MIN_BEHAV_TRIAS = 400
MIN_RT = 0.08  # 0.08  # Float (s) or None
# Basically, quality metric on the stability of a single unit. Should have 1 metric per neuron
QC_CRITERIA = 3/3  # 3 / 3  # In {None, 1/3, 2/3, 3/3}

# decoder and null distribution
ESTIMATOR = sklm.LogisticRegression #sklm.Lasso  # Must be in keys of strlut above
ESTIMATOR_KWARGS = {'penalty': 'l1', 'solver':'saga', 'tol': 0.0001, 'max_iter': 10000, 'fit_intercept': True}#'penalty': 'l1', 'solver':'saga', 
SCORE = 'accuracy' #r2 or accuracy
N_PSEUDO = 100

NO_UNBIAS = False
SHUFFLE = True
COMPUTE_NEUROMETRIC = False #True if TARGET == 'signcont' else False
FORCE_POSITIVE_NEURO_SLOPES = False
BALANCED_WEIGHT = True # seems to work better with BALANCED_WEIGHT=False
HPARAM_GRID = {'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])}
ESTIMATORSTR = strlut[ESTIMATOR]  
if ESTIMATORSTR == 'Logistic':
    HPARAM_GRID = {'C': np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])}
DOUBLEDIP = False
SAVE_BINNED = False  # Debugging parameter, not usually necessary
COMPUTE_NEURO_ON_EACH_FOLD = False  # if True, expect a script that is 5 times slower
ADD_TO_SAVING_PATH = '20eidV0'

# ValueErrors and NotImplementedErrors
if TARGET not in ['prior','signcont','pLeft','choice','feedback']:
    raise NotImplementedError('this TARGET is not supported or stable yet')

if MODEL not in list(dut.modeldispatcher.keys()):
    raise NotImplementedError('this MODEL is not supported or stable yet')

if COMPUTE_NEUROMETRIC and TARGET != 'signcont':
    raise ValueError('the target should be signcont to compute neurometric curves')

if ESTIMATORSTR == 'Logistic' and TARGET == 'choice':
    MASK_DATA = lambda x: (np.array(x)==-1)|(np.array(x)==1)
    TRANSFORM_DATA = lambda x: np.array((x+1)/2, dtype=int)
elif ESTIMATORSTR == 'Logistic' and TARGET == 'pLeft':
    MASK_DATA = lambda x: (np.array(x)==0.2)|(np.array(x)==0.8)
    TRANSFORM_DATA = lambda x: np.array((x-0.2)/0.6, dtype=int)
elif ESTIMATORSTR == 'Logistic' and TARGET == 'feedback':
    MASK_DATA = lambda x: (np.array(x)==-1)|(np.array(x)==1)
    TRANSFORM_DATA = lambda x: np.array((x+1)/2, dtype=int)
else:
    MASK_DATA = lambda x: (np.array(x)==np.array(x))
    TRANSFORM_DATA = lambda x: x
    
fit_metadata = {
    'criterion': SESS_CRITERION,
    'target': TARGET,
    'control_features': CONTROL_FEATURES,
    'model_type': dut.modeldispatcher[MODEL],
    'model_score': SCORE,
    'modelfit_path': MODELFIT_PATH,
    'output_path': OUTPUT_PATH,
    'align_time': ALIGN_TIME,
    'time_window': TIME_WINDOW,
    'estimator': ESTIMATORSTR,
    'n_pseudo': N_PSEUDO,
    'min_units': MIN_UNITS,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'qc_criteria': QC_CRITERIA,
    'date': DATE,
    'shuffle': SHUFFLE,
    'no_unbias': NO_UNBIAS,
    'hyperparameter_grid': HPARAM_GRID,
    'save_binned': SAVE_BINNED,
    'balanced_weight': BALANCED_WEIGHT,
    'double_dip': DOUBLEDIP,
    'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
    'compute_neurometric': COMPUTE_NEUROMETRIC
}


# %% Define helper functions
    

def save_region_results(fit_result, pseudo_results, 
                        subject, eid, probe, region, N):#
    decodingdetailsfolder = Path(OUTPUT_PATH).joinpath(dut.decoding_details(TARGET,MODEL,SCORE,
                                                      ESTIMATORSTR,
                                                      ALIGN_TIME,
                                                      CONTROL_FEATURES,
                                                      N_PSEUDO,TIME_WINDOW,
                                                      ADD_TO_SAVING_PATH))
    subjectfolder = decodingdetailsfolder.joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    for folder in [decodingdetailsfolder, subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    fn = '_'.join([DATE, region]) + '.pkl'
    fw = open(probefolder.joinpath(fn), 'wb')
    outdict = {'fit': fit_result, 'pseudosessions': pseudo_results,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region, 'N_units': N}
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


def fit_eid(eid, sessdf):
    one = ONE()  # mode='local'
    atlas = AllenAtlas()

    estimator = ESTIMATOR #(**ESTIMATOR_KWARGS)

    subject = sessdf.xs(eid, level='eid').index[0]
    subjeids = sessdf.xs(subject, level='subject').index.unique()
    brainreg = dut.BrainRegions()
    behavior_data = mut.load_session(eid, one=one)
    pLeft_vec = np.array(behavior_data['probabilityLeft'])
    try:
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, beh_data=behavior_data,
                                  one=one)
    except ValueError:
        print('Model not fit.')
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, one=one)
    
    fvecs = [dut.compute_target(feature, subject, subjeids, eid, MODELFIT_PATH,
                              modeltype=MODEL, beh_data=behavior_data,
                              one=one) for feature in CONTROL_FEATURES]
    
    try:
        trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])
        if len(trialsdf) != len(tvec):
            raise IndexError
    except IndexError:
        raise IndexError('Problem in the dimensions of dataframe of session')
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf['goCue_times']
    mask = trialsdf[ALIGN_TIME].notna()
    mask = mask & MASK_DATA(tvec)
    if NO_UNBIAS:
        mask = mask & (trialsdf.probabilityLeft != 0.5).values
    if MIN_RT is not None:
        mask = mask & (~(trialsdf.react_times < MIN_RT)).values

    nb_trialsdf = trialsdf[mask]
    msub_tvec = TRANSFORM_DATA(tvec[mask])
    fvecs = [fvec[mask] for fvec in fvecs]
    

    # doubledipping
    if DOUBLEDIP:
        msub_tvec = msub_tvec - np.mean(msub_tvec)

    filenames = []
    if len(msub_tvec) <= MIN_BEHAV_TRIAS:
        return filenames

    print(f'Working on eid : {eid}')
    for probe in tqdm(sessdf.loc[subject, eid].probe, desc='Probe: ', leave=False):
        # load_spike_sorting_fast
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                    one=one,
                                                                    probe=probe,
                                                                    brain_atlas=atlas,
                                                                    dataset_types=['spikes.depths', 'spikes.amps'],
                                                                    aligned=True)

        beryl_reg = dut.remap_region(clusters[probe].atlas_id, br=brainreg)
        if QC_CRITERIA:
            metrics = pd.DataFrame.from_dict(quick_unit_metrics(spikes[probe].clusters,
                                                                spikes[probe].times,
                                                                spikes[probe].amps,
                                                                spikes[probe].depths,
                                                                cluster_ids=np.arange(beryl_reg.size)))
            try:
                metrics_verif = clusters[probe].metrics
                if beryl_reg.shape[0] == len(metrics_verif):
                    if not np.all(((metrics_verif.label - metrics.label) < 1e-10) + metrics_verif.label.isna()):
                        raise ValueError('there is a problem in the metric computations')
            except AttributeError:
                pass
            qc_pass = (metrics.label >= QC_CRITERIA).values
            if beryl_reg.shape[0] != len(qc_pass):
                raise IndexError('Shapes of metrics and number of clusters '
                                 'in regions dont match')
        else:
            qc_pass = np.ones_like(beryl_reg, dtype=bool)
        regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        for region in tqdm(regions, desc='Region: ', leave=False):
            reg_mask = beryl_reg == region
            reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            N_units = len(reg_clu_ids)
            if N_units < MIN_UNITS:
                continue
            # or get_spike_count_in_bins
            if np.any(np.isnan(nb_trialsdf[ALIGN_TIME])):
                # if this happens, verify scrub of NaN values in all aign times before get_spike_counts_in_bins
                raise ValueError('this should not happen')
            intervals = np.vstack([nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[0],
                                   nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[1]]).T
            spikemask = np.isin(spikes[probe].clusters, reg_clu_ids)
            regspikes = spikes[probe].times[spikemask]
            regclu = spikes[probe].clusters[spikemask]
            
            binned_neurons, _ = get_spike_counts_in_bins(regspikes, regclu,
                                                 intervals)
            
            # construct features used for decoding:
            #   often neural activity in the shape (n_neurons, n_trials), but
            #   can include additional features according to DECODING_FEATURES
            all_features = [binned_neurons[i,:] for i in range(binned_neurons.shape[0])]
#             for fvec in fvecs:
#                 all_features.append(fvec)
            try:
                binned = np.vstack(all_features)
            except ValueError:
                raise ValueError('decoding features may have different lengths')
            
            # doubledipping
            msub_binned = binned.T
            if DOUBLEDIP:
                msub_binned = binned.T - np.mean(binned.T, axis=0) # binned.T.astype(int)

            if len(msub_binned.shape) > 2:
                raise ValueError('Multiple bins are being calculated per trial,'
                                 'may be due to floating point representation error.'
                                 'Check window.')
            fit_result = dut.regress_target(msub_tvec, msub_binned, estimator,
                                            estimator_kwargs=ESTIMATOR_KWARGS,
                                            hyperparam_grid=HPARAM_GRID,
                                            save_binned=SAVE_BINNED, shuffle=SHUFFLE,
                                            balanced_weight=BALANCED_WEIGHT,
                                            control_features=fvecs,
                                            SCORE=SCORE)

            fit_result['mask'] = mask
            fit_result['pLeft_vec'] = pLeft_vec

            # neurometric curve
            if COMPUTE_NEUROMETRIC:
                fit_result['full_neurometric'], fit_result['fold_neurometric'] = \
                    get_neurometric_parameters(fit_result, nb_trialsdf.reset_index(), one,
                                               compute_on_each_fold=COMPUTE_NEURO_ON_EACH_FOLD,
                                               force_positive_neuro_slopes=FORCE_POSITIVE_NEURO_SLOPES)
            else:
                fit_result['full_neurometric'] = None
                fit_result['fold_neurometric'] = None

            pseudo_results = []
            for _ in tqdm(range(N_PSEUDO), desc='Pseudo num: ', leave=False):
                pseudosess = generate_pseudo_session(trialsdf)
                pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                                      MODELFIT_PATH, modeltype=MODEL,
                                                      beh_data=pseudosess, one=one)
                msub_pseudo_tvec = TRANSFORM_DATA(pseudo_tvec[mask])   
                # doubledipping
                if DOUBLEDIP:
                    msub_pseudo_tvec = msub_pseudo_tvec - np.mean(msub_pseudo_tvec)

                pseudo_result = dut.regress_target(msub_pseudo_tvec, msub_binned, estimator,
                                                   estimator_kwargs=ESTIMATOR_KWARGS,
                                                   hyperparam_grid=HPARAM_GRID,
                                                   save_binned=SAVE_BINNED, shuffle=SHUFFLE,
                                                   balanced_weight=BALANCED_WEIGHT,
                                                   SCORE=SCORE)

                # neurometric curve
                if COMPUTE_NEUROMETRIC:
                    pseudo_result['full_neurometric'], pseudo_result['fold_neurometric'] = \
                        get_neurometric_parameters(pseudo_result, pseudosess[mask].reset_index(),
                                                   one, compute_on_each_fold=COMPUTE_NEURO_ON_EACH_FOLD,
                                                   force_positive_neuro_slopes=FORCE_POSITIVE_NEURO_SLOPES)
                else:
                    pseudo_result['full_neurometric'] = None
                    pseudo_result['fold_neurometric'] = None
                
                pseudo_result['pLeft_vec'] = pLeft_vec

                pseudo_results.append(pseudo_result)
            filenames.append(save_region_results(fit_result, pseudo_results, subject,
                                                 eid, probe, region, N_units))

    return filenames