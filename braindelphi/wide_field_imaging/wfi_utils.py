import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
import numpy
import torch



def decompose_downsampled_files(path_to_wfi):
    downsampled = numpy.load(path_to_wfi + '/downsampled_ims-%s.npy' % path_to_wfi.split('-')[-1], allow_pickle=True)
    for i_sess in tqdm(range(downsampled.shape[0])):
        np.save(path_to_wfi + '/downsampled_ims-'+path_to_wfi.split('-')[-1]+'-'+str(i_sess)+'.npy', downsampled[i_sess] )

'''
subjects = glob.glob('CSK-im-*')
for subject_path in subjects:
    print(subject_path)
    decompose_downsampled_files(subject_path)
'''

def load_wfi(path_to_wfi, sessid=-1):
    
    if sessid == -1 :
        downsampled = numpy.load(path_to_wfi + '/downsampled_ims-%s.npy' % path_to_wfi.split('-')[-1], allow_pickle=True)
        downsampled = downsampled[sessid]
    else :
        downsampled = numpy.load(path_to_wfi + '/downsampled_ims-'+path_to_wfi.split('-')[-1]+'-'+str(sessid)+'.npy', allow_pickle=True)

    behavior = numpy.load(path_to_wfi + '/behavior.npy', allow_pickle=True)
    ds_atlas = numpy.load(path_to_wfi + '/ds_atlas.npy', allow_pickle=True)
    frame_df = numpy.load(path_to_wfi + '/frame_df.npy', allow_pickle=True)
    ds_map = numpy.load(path_to_wfi + '/ds_map.npy', allow_pickle=True)
    return downsampled, behavior, ds_atlas, ds_map, frame_df


def load_wfi_session(path_to_wfi, sessid):
    downsampled, behavior, ds_atlas, ds_map, frame_df = load_wfi(path_to_wfi, sessid)
    atlas = pd.read_csv('ccf_regions.csv')
    for k in range(behavior.size):
        behavior[k]['session_id'] = k
        behavior[k]['session_to_decode'] = True if k == sessid else False
    sessiondf = pd.concat(behavior, axis=0).reset_index(drop=True)
    sessiondf['subject'] = 'wfi%i' % int(path_to_wfi.split('-')[-1])
    sessiondf = sessiondf[['choice', 'stimOn_times', 'feedbackType', 'feedback_times', 'contrastLeft', 'goCue_times',
                           'contrastRight', 'probabilityLeft', 'session_to_decode', 'subject', 'signedContrast',
                           'session_id', 'firstMovement_times']]
    sessiondf = sessiondf.assign(stim_side=(sessiondf.choice * (sessiondf.feedbackType == 1) -
                                            sessiondf.choice * (sessiondf.feedbackType == -1)))
    sessiondf['eid'] = sessiondf['session_id'].apply(lambda x: ('wfi' + str(int(path_to_wfi.split('-')[-1]))
                                                                + 's' + str(x)))
    # downsampled = downsampled[sessid]
    frame_df = frame_df[sessid]
    wideFieldImaging_dict = {'activity': downsampled, 'timings': frame_df, 'regions': ds_atlas, 'atlas': atlas}
    return sessiondf, wideFieldImaging_dict



def format_resultsdf(resultsdf,target="prior",correction='median',additional_df=False):

    '''
    Generic function to turn the dataframe outputted by the decoding pipeline into a format usable for analysis and plots.

    Parameters
    resultsdf (DataFrame) : dataframe which contain the results of the decoding pipeline, output of file 06_slurm_format.py
    target (str) : what was the target of the decoding ( 'prior' or 'choice')
    correction (str) : Which statistic of the null distribution to use to correct the results statistics ('mean' or 'median')
    additional_df (Boolean) : Wether to return dataframes with more information about each regression of not
    
    Returns
    region_results (Dataframe) : dataframe containing decoding corrected R^2 score for every region
    '''
    
    # mean over the runs to remove the variability du to the cross-validation
    df_subject = resultsdf.groupby(['eid','probe','region','pseudo_id']).mean()
    
    # isolate the true sessions regressions
    df_true = df_subject[ df_subject.index.get_level_values('pseudo_id') == -1]

    # isolate the pseudo sessions regressions
    df_pseudo = df_subject[ df_subject.index.get_level_values('pseudo_id') != -1]

    # compute the median of pseudo-sessions results over the pseudo_id for each regression
    if correction == 'median':
        df_pseudo_correction = df_pseudo.groupby(['eid','probe','region']).median()
    else :
        df_pseudo_correction = df_pseudo.groupby(['eid','probe','region']).mean()
    df_pseudo_correction = df_pseudo_correction.add_prefix('pseudo_')
    
    # join pseudo regressions median results to each corresponding true regression results
    df_sess_reg = df_true.join(df_pseudo_correction, how='outer')

    # for each regression, get the corrected results for r2 test
    df_sess_reg['corrected_R2_test'] = df_sess_reg["R2_test"] - df_sess_reg["pseudo_R2_test"]

    # we compute 95% quantile for each sess-reg
    

    # average corrected results over all regressions for each region
    df_reg = df_sess_reg.groupby('region').mean()
    
    # compute the 95% quantile for R2 
    # first average the shift over all regressions for each region for each pseudosession
    df_reg_pseudo = df_subject.groupby(['region','pseudo_id']).mean()
    df_pseudo_R2_reg = df_pseudo['R2_test'].groupby(['region','pseudo_id']).mean()

    # compute 95% quantile for each region to assess significativity
    df_pseudo_quant = df_pseudo_R2_reg.groupby('region').quantile(0.95)
    df_pseudo_quant = df_pseudo_quant.rename('pseudo_95_quantile')

    region_results = df_reg.join(df_pseudo_quant, how='outer')

    df_pseudo_sess = df_pseudo.groupby(['region','pseudo_id','eid']).mean()
    
    df_sess_reg['pseudo_distrib'] = df_pseudo_sess[['R2_test','eid']].groupby('eid').agg(list)

    df_sess_reg['tvalue'] = df_sess_reg.apply(lambda x : scipy.stats.ttest_1samp(x.pseudo_distrib, x.tvalue).statistic)
    
    if additional_df :
        # for fine exploration of the data
        return region_results,df_sess_reg,df_pseudo,df_reg_pseudo
    else :
        return region_results
        
def format_data(data):
    
    stim_side = data['stim_side']
    stimuli = np.nan_to_num(data['contrastLeft']) - np.nan_to_num(data['contrastRight'])
    if 'choice' in data.keys():
        actions = data['choice']
    else:
        actions = pd.Series(dtype='float64').reindex_like(stim_side)
    pLeft_oracle = data['probabilityLeft']
    return np.array(stim_side), np.array(stimuli), np.array(actions), np.array(pLeft_oracle)
    
def optimal_Bayesian(act, side):
    '''
    Generates the optimal prior
    Params:
        act (array of shape [nb_sessions, nb_trials]): action performed by the mice of shape
        side (array of shape [nb_sessions, nb_trials]): stimulus side (-1 (right), 1 (left)) observed by the mice
    Output:
        prior (array of shape [nb_sessions, nb_chains, nb_trials]): prior for each chain and session
    '''
    act = torch.from_numpy(act)
    side = torch.from_numpy(side)
    lb, tau, ub, gamma = 20, 60, 100, 0.8
    nb_blocklengths = 100
    nb_typeblocks = 3
    eps = torch.tensor(1e-15)

    alpha = torch.zeros([act.shape[-1], nb_blocklengths, nb_typeblocks])
    alpha[0, 0, 1] = 1
    alpha = alpha.reshape(-1, nb_typeblocks * nb_blocklengths)
    h = torch.zeros([nb_typeblocks * nb_blocklengths])

    # build transition matrix
    b = torch.zeros([nb_blocklengths, nb_typeblocks, nb_typeblocks])
    b[1:][:, 0, 0], b[1:][:, 1, 1], b[1:][:, 2, 2] = 1, 1, 1  # case when l_t > 0
    b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = 1, 1, 1. / 2  # case when l_t = 1
    n = torch.arange(1, nb_blocklengths + 1)
    ref = torch.exp(-n / tau) * (lb <= n) * (ub >= n)
    torch.flip(ref.double(), (0,))
    hazard = torch.cummax(
        ref / torch.flip(torch.cumsum(torch.flip(ref.double(), (0,)), 0) + eps, (0,)), 0)[0]
    l = torch.cat(
        (torch.unsqueeze(hazard, -1),
         torch.cat((torch.diag(1 - hazard[:-1]), torch.zeros(nb_blocklengths - 1)[None]), axis=0)),
        axis=-1)  # l_{t-1}, l_t
    transition = eps + torch.transpose(l[:, :, None, None] * b[None], 1, 2).reshape(
        nb_typeblocks * nb_blocklengths, -1)

    # likelihood
    lks = torch.hstack([
        gamma * (side[:, None] == -1) + (1 - gamma) * (side[:, None] == 1),
        torch.ones_like(act[:, None]) * 1. / 2,
        gamma * (side[:, None] == 1) + (1 - gamma) * (side[:, None] == -1)
    ])
    to_update = torch.unsqueeze(torch.unsqueeze(act.not_equal(0), -1), -1) * 1

    for i_trial in range(act.shape[-1]):
        # save priors
        if i_trial > 0:
            alpha[i_trial] = torch.sum(torch.unsqueeze(h, -1) * transition, axis=0) * to_update[i_trial - 1] \
                             + alpha[i_trial - 1] * (1 - to_update[i_trial - 1])
        h = alpha[i_trial] * lks[i_trial].repeat(nb_blocklengths)
        h = h / torch.unsqueeze(torch.sum(h, axis=-1), -1)

    predictive = torch.sum(alpha.reshape(-1, nb_blocklengths, nb_typeblocks), 1)
    Pis = predictive[:, 0] * gamma + predictive[:, 1] * 0.5 + predictive[:, 2] * (1 - gamma)

    return 1 - Pis



def select_widefield_imaging_regions(neural_dict, region, kwargs):
    """Select pixels based on brain region."""
    region_labels = []
    reg_lab = neural_dict['atlas'][neural_dict['atlas'].acronym.isin(region).values].label.values.squeeze()
    if 'left' in kwargs['wfi_hemispheres']:
        region_labels.append(reg_lab)
    if 'right' in kwargs['wfi_hemispheres']:
        region_labels.append(-reg_lab)

    reg_mask = np.isin(neural_dict['regions'], region_labels)
    return reg_mask

def preprocess_widefield_imaging(neural_dict, reg_mask, kwargs):
    frames_idx = np.sort(
        neural_dict['timings'][kwargs['align_time']].values[:, None] +
        np.arange(kwargs['wfi_nb_frames_start'], kwargs['wfi_nb_frames_end'] + 1),
        axis=1,
    )
    binned = np.take(neural_dict['activity'][:, reg_mask],
                    frames_idx,
                    axis=0)
    binned = list(binned.reshape(binned.shape[0], -1)[:, None])
    return binned