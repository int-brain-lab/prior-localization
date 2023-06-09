import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
import numpy
import torch
import wfield


def preprocess_widefield_imaging(neural_dict, reg_mask, **kwargs):
    frames_idx = np.sort(
        neural_dict['timings'][kwargs['align_time']].values[:, None] +
        np.arange(kwargs['wfi_nb_frames_start'], kwargs['wfi_nb_frames_end'] + 1),
        axis=1,
    )
    binned = np.take(neural_dict['activity'], # [:, reg_mask]
                     frames_idx,
                     axis=0)
    binned = binned[:, :, reg_mask]
    if kwargs['wfi_average_over_frames']:
        binned = binned.mean(axis=1, keepdims=True)
    binned = list(binned.reshape(binned.shape[0], -1)[:, None])
    return binned


def select_ephys_regions(neural_dict, beryl_reg, region, **kwargs):
    """Select units based on QC criteria and brain region."""
    qc_pass = np.asarray(neural_dict['clusters']['label'] >= kwargs['qc_criteria'])
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
    return reg_clu_ids


def get_bery_reg_wfi(neural_dict, **kwargs):
    uniq_regions = np.unique(neural_dict['regions'])
    if 'left' in kwargs['wfi_hemispheres'] and 'right' in kwargs['wfi_hemispheres']:
        return neural_dict['atlas'].acronym[np.array([k in np.abs(uniq_regions)
                                                      for k in neural_dict['atlas'].label])]
    elif 'left' in kwargs['wfi_hemispheres']:
        return neural_dict['atlas'].acronym[np.array([k in np.abs(uniq_regions[uniq_regions > 0])
                                                      for k in neural_dict['atlas'].label])]
    elif 'right' in kwargs['wfi_hemispheres']:
        return neural_dict['atlas'].acronym[np.array([k in np.abs(uniq_regions[uniq_regions < 0])
                                                      for k in neural_dict['atlas'].label])]
    else:
        raise ValueError('there is a problem in the wfi_hemispheres argument')


def select_widefield_imaging_regions(neural_dict, region, **kwargs):
    """Select pixels based on brain region."""
    region_labels = []
    reg_lab = neural_dict['atlas'][neural_dict['atlas'].acronym.isin(region).values].label.values
    if 'left' in kwargs['wfi_hemispheres']:
        region_labels.extend(reg_lab)
    if 'right' in kwargs['wfi_hemispheres']:
        region_labels.extend(-reg_lab)

    reg_mask = np.isin(neural_dict['regions'], region_labels)
    return reg_mask

def load_wfi_session(path_to_wfi_subject, sessid, hemispheres, path_to_wfi, keep_all_session=False):
    downsampled, behavior, ds_atlas, ds_map, frame_df = load_wfi(path_to_wfi_subject)
    atlas = pd.read_csv(path_to_wfi + '/ccf_regions.csv')
    for k in range(behavior.size):
        behavior[k]['eid'] = k
        behavior[k]['session_to_decode'] = True if k == sessid else False
    sessiondf = pd.concat(behavior, axis=0).reset_index(drop=True)
    if not keep_all_session:
        sessiondf = sessiondf[sessiondf.session_to_decode]
    if 'left' not in hemispheres:  # if 'left' is not in hemispheres, only keep regions with negative labels (right hem)
        ds_atlas = (ds_atlas < 0) * ds_atlas
    if 'right' not in hemispheres:  # if 'right' is not in hemispheres, only keep regions with positive labels (left hem)
        ds_atlas = (ds_atlas > 0) * ds_atlas
    sessiondf['subject'] = 'wfi%i' % int(path_to_wfi_subject.split('-')[-1])
    sessiondf = sessiondf[['choice', 'stimOn_times', 'feedbackType', 'feedback_times', 'contrastLeft', 'goCue_times',
                           'contrastRight', 'probabilityLeft', 'session_to_decode', 'subject', 'signedContrast',
                           'eid', 'firstMovement_times']]
    sessiondf = sessiondf.assign(stim_side=(sessiondf.choice * (sessiondf.feedbackType == 1) -
                                            sessiondf.choice * (sessiondf.feedbackType == -1)))
    sessiondf['eid'] = sessiondf['eid'].apply(lambda x: ('wfi' + str(int(path_to_wfi_subject.split('-')[-1]))
                                                                + 's' + str(x)))
    downsampled = downsampled[sessid]
    frame_df = frame_df[sessid]
    wideFieldImaging_dict = {'activity': downsampled, 'timings': frame_df,
                             'regions': ds_atlas, 'atlas': atlas[['acronym', 'name', 'label']]}

    if sessiondf.eid[sessiondf.session_to_decode].unique().size > 1:
        raise ValueError('there is a problem in the code')
    if sessiondf.subject[sessiondf.session_to_decode].unique().size > 1:
        raise ValueError('there is a problem in the code')

    metadata = {'eid': sessiondf.eid[sessiondf.session_to_decode].unique()[0],
                'subject': sessiondf.subject[sessiondf.session_to_decode].unique()[0],}

    return sessiondf, wideFieldImaging_dict, metadata


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
    else :
        downsampled = numpy.load(path_to_wfi + '/downsampled_ims-'+path_to_wfi.split('-')[-1]+'-'+str(sessid)+'.npy', allow_pickle=True)

    behavior = numpy.load(path_to_wfi + '/behavior.npy', allow_pickle=True)
    ds_atlas = numpy.load(path_to_wfi + '/ds_atlas.npy', allow_pickle=True)
    frame_df = numpy.load(path_to_wfi + '/frame_df.npy', allow_pickle=True)
    ds_map = numpy.load(path_to_wfi + '/ds_map.npy', allow_pickle=True)
    return downsampled, behavior, ds_atlas, ds_map, frame_df


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

# Taken from https://github.com/cskrasniak/wfield/blob/master/wfield/analyses.py
def downsample_atlas(atlas, pixelSize=20, mask=None):
    """
    Downsamples the atlas so that it can be matching to the downsampled images. if mask is not provided
    then just the atlas is used. pixelSize must be a common divisor of 540 and 640
    """
    if not mask:
        mask = atlas != 0
    downsampled_atlas = np.zeros((int(atlas.shape[0] / pixelSize), int(atlas.shape[1] / pixelSize)))
    for top in np.arange(0, 540, pixelSize):
        for left in np.arange(0, 640, pixelSize):
            useArea = (np.array([np.arange(top, top + pixelSize)] * pixelSize).flatten(),
                       np.array([[x] * pixelSize for x in range(left, left + pixelSize)]).flatten())
            u_areas, u_counts = np.unique(atlas[useArea], return_counts=True)
            if np.sum(mask[useArea] != 0) < .5:
                # if more than half of the pixels are outside of the brain, skip this group of pixels
                continue
            else:
                spot_label = u_areas[np.argmax(u_counts)]
                downsampled_atlas[int(top / pixelSize), int(left / pixelSize)] = spot_label
    return downsampled_atlas.astype(int)


def spatial_down_sample(stack, pixelSize=20):
    """
    Downsamples the whole df/f video for a session to a manageable size, best are to do a 10x or
    20x downsampling, this makes many tasks more manageable on a desktop.
    """
    mask = stack.U_warped != 0
    mask = mask.mean(axis=2)
    try:
        downsampled_im = np.zeros((stack.SVT.shape[1],
                                   int(stack.U_warped.shape[0] / pixelSize),
                                   int(stack.U_warped.shape[1] / pixelSize)))
    except:
        print('Choose a downsampling amount that is a common divisor of 540 and 640')
    for top in np.arange(0, 540, pixelSize):
        for left in np.arange(0, 640, pixelSize):
            useArea = (np.array([np.arange(top, top + pixelSize)] * pixelSize).flatten(),
                       np.array([[x] * pixelSize for x in range(left, left + pixelSize)]).flatten())
            if np.sum(mask[useArea] != 0) < .5:
                # if more than half of the pixels are outside of the brain, skip this group of pixels
                continue
            else:
                spot_activity = stack.get_timecourse(useArea).mean(axis=0)
                downsampled_im[:, int(top / pixelSize), int(left / pixelSize)] = spot_activity
    return downsampled_im

def prepare_widefield_data(eid, one, corrected=True):
    if corrected:
        SVT = one.load_dataset(eid, 'widefieldSVT.haemoCorrected.npy')
    else:
        SVT = one.load_dataset(eid, 'widefieldSVT.uncorrected.npy')
    U = one.load_dataset(eid, 'widefieldU.images.npy')
    times = one.load_dataset(eid, 'imaging.times.npy')
    channels = one.load_dataset(eid, 'imaging.imagingLightSource.npy')
    channel_info = one.load_dataset(eid, 'imagingLightSource.properties.htsv', download_only=True)
    channel_info = pd.read_csv(channel_info)
    lmark_file = one.load_dataset(eid, 'widefieldLandmarks.dorsalCortex.json', download_only=True)
    landmarks = wfield.load_allen_landmarks(lmark_file)
    # If haemocorrected need to take timestamps that correspond to functional channel
    functional_channel = 470
    functional_chn = channel_info.loc[channel_info['wavelength'] == functional_channel]['channel_id'].values[0]
    times = times[channels == functional_chn]

    # Align the image stack to Allen reference
    stack = wfield.SVDStack(U, SVT)
    stack.set_warped(True, M=landmarks['transform'])

    # Load in the Allen atlas
    atlas, area_names, mask = wfield.atlas_from_landmarks_file(lmark_file, do_transform=False)
    ccf_regions, _, _ = wfield.allen_load_reference('dorsal_cortex')
    ccf_regions = ccf_regions[['acronym', 'name', 'label']]

    # Create a 3d mask of the brain outline
    mask3d = wfield.mask_to_3d(mask, shape=np.roll(stack.U_warped.shape, 1))
    # Set pixels outside the brain outline to zero
    stack.U_warped[~mask3d.transpose([1, 2, 0])] = 0
    # Do the same to the Allen image
    atlas[~mask] = 0

    # Downsample the images
    downsampled_atlas = downsample_atlas(atlas, pixelSize=10)
    downsampled_stack = spatial_down_sample(stack, pixelSize=10)


    trials = one.load_object(eid, 'trials')
    trials = trials.to_df()
    frames = pd.DataFrame()
    for key in trials.keys():
        if 'times' in key:
            idx = np.searchsorted(times, trials[key].values).astype(np.float64)
            idx[np.isnan(trials[key].values)] = np.nan
            frames[key] = idx
        else:
            frames[key] = trials[key].values

    # remove last trial as this is detected wrong
    frames = frames[:-1]

    neural_activity = {}
    neural_activity['activity'] = downsampled_stack
    neural_activity['timings'] = frames
    neural_activity['regions'] = downsampled_atlas
    neural_activity['atlas'] = ccf_regions


    return neural_activity