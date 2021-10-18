# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import alf
from os.path import join
from brainbox.numerical import ismember
from ibllib.atlas import BrainRegions
from oneibl.one import ONE


def paths():
    """
    Make a file in the root of the repository called 'paths.py' with in it:

    DATA_PATH = '/path/to/Flatiron/data'
    FIG_PATH = '/path/to/save/figures'
    SAVE_PATH = '/path/to/save/data'

    """
    from paths import DATA_PATH, FIG_PATH, SAVE_PATH
    return DATA_PATH, FIG_PATH, SAVE_PATH


def figure_style(font_scale=2, despine=False, trim=True):
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=font_scale)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if despine is True:
        sns.despine(trim=trim)
        plt.tight_layout()


def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    if one is None:
        one = ONE()
    trials = pd.DataFrame()
    if laser_stimulation:
        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice'],
         trials['feedback_times'], trials['firstMovement_times'], trials['laser_stimulation'],
         trials['laser_probability']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 'trials.feedback_times', 'trials.firstMovement_times',
                                                 '_ibl_trials.laser_stimulation',
                                                 '_ibl_trials.laser_probability'])
        if trials.shape[0] == 0:
            return
        if trials.loc[0, 'laser_stimulation'] is None:
            trials = trials.drop(columns=['laser_stimulation'])
        if trials.loc[0, 'laser_probability'] is None:
            trials = trials.drop(columns=['laser_probability'])
    else:
       (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice'], trials['firstMovement_times'],
         trials['feedback_times']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 'trials.firstMovement_times',
                                                 'trials.feedback_times'])
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1
    if 'firstMovement_times' in trials.columns.values:
        trials['reaction_times'] = trials['firstMovement_times'] - trials['goCue_times']
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials


def check_trials(trials):
    trials = trials.reset_index(drop=True)
    if trials is None:
        print('trials is None type')
        return False
    if trials.probabilityLeft is None:
        print('trials.probabilityLeft is None type')
        return False
    if len(trials.probabilityLeft[0].shape) > 0:
        print('trials.probabilityLeft is an array of tuples')
        return False
    if (('stimOn_times' not in trials.columns)
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        print('stimOn_times do not match with probabilityLeft')
        return False
    return True


def remap(ids, source='Allen', dest='Beryl', output='acronym'):
    br = BrainRegions()
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']


def query_sessions(selection='all', return_subjects=False):
    one = ONE()
    if selection == 'all':
        # Query all ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all sessions with at least one alignment
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0')
    elif selection == 'resolved':
        # Query all sessions with resolved alignment
         ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_resolved,True')
    elif selection == 'aligned-behavior':
        # Query sessions with at least one alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    elif selection == 'resolved-behavior':
        # Query sessions with resolved alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_resolved,True,'
                               'session__extended_qc__behavior,1')
    else:
        ins = []

    # Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    eids, ind_unique = np.unique(all_eids, return_index=True)
    subjects = all_subjects[ind_unique]
    probes = []
    for i, eid in enumerate(eids):
        probes.append(all_probes[[s == eid for s in all_eids]])
    if return_subjects:
        return eids, probes, subjects
    else:
        return eids, probes


def sessions_with_region(acronym):
    from oneibl.one import ONE
    one = ONE()

    # Query sessions with at least one channel in the specified region
    sessions = one.alyx.rest('sessions', 'list', atlas_acronym=acronym,
                        task_protocol='_iblrig_tasks_ephysChoiceWorld',
                        dataset_types = ['spikes.times', 'trials.probabilityLeft'],
                        project='ibl_neuropixel_brainwide')
    return sessions


def combine_layers_cortex(regions):
    """
    Combine all layers of cortex
    """
    remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
    for i, region in enumerate(regions):
        if region[:2] == 'CA':
            continue
        for j, char in enumerate(remove):
            regions[i] = regions[i].replace(char, '')
    return regions


def get_parent_region_name(acronyms):
    brainregions = BrainRegions()
    parent_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regid = brainregions.id[np.argwhere(brainregions.acronym == acronym)]
            ancestors = brainregions.ancestors(regid)
            targetlevel = 6
            if sum(ancestors.level == targetlevel) == 0:
                parent_region_names.append(ancestors.name[-1])
            else:
                parent_region_names.append(ancestors.name[np.argwhere(
                                                ancestors.level == targetlevel)[0, 0]])
        except IndexError:
            parent_region_names.append(acronym)
    if len(parent_region_names) == 1:
        return parent_region_names[0]
    else:
        return parent_region_names


def get_children_region_names(acronyms, return_full_name=False):
    br = BrainRegions()
    children_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regid = br.id[np.argwhere(br.acronym == acronym)]
            descendants = br.descendants(regid)
            targetlevel = 8
            if sum(descendants.level == targetlevel) == 0:
                if return_full_name:
                    children_region_names.append(descendants.name[-1])
                else:
                    children_region_names.append(descendants.acronym[-1])
            else:
                if return_full_name:
                    children_region_names.append(descendants.name[
                        (descendants.level == targetlevel) & (descendants.id > 0)])
                else:
                    children_region_names.append(descendants.acronym[
                        (descendants.level == targetlevel) & (descendants.id > 0)])
        except IndexError:
            children_region_names.append(acronym)
    if len(children_region_names) == 1:
        return children_region_names[0]
    else:
        return children_region_names


def get_full_region_name(acronyms):
    brainregions = BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


def get_eid_list():
    eids = np.array([
            '465c44bd-2e67-4112-977b-36e1ac7e3f8c', 'db4df448-e449-4a6f-a0e7-288711e7a75a',
            '158d5d35-a2ab-4a76-87b0-51048c5d5283', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
            'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', '36280321-555b-446d-9b7d-c2e17991e090',
            'bb6a5aae-2431-401d-8f6a-9fdd6de655a9', 'a71175be-d1fd-47a3-aa93-b830ea3634a1',
            '2199306e-488a-40ab-93cb-2d2264775578', '4d8c7767-981c-4347-8e5e-5d5fffe38534',
            '30e5937e-e86a-47e6-93ae-d2ae3877ff8e', 'e535fb62-e245-4a48-b119-88ce62a6fe67',
            '8435e122-c0a4-4bea-a322-e08e8038478f', 'b39752db-abdb-47ab-ae78-e8608bbf50ed',
            'fa704052-147e-46f6-b190-a65b837e605e', '3dd347df-f14e-40d5-9ff2-9c49f84d2157',
            'e5fae088-ed96-4d9b-82f9-dfd13c259d52', 'e49d8ee7-24b9-416a-9d04-9be33b655f40',
            'f9860a11-24d3-452e-ab95-39e199f20a93', 'b658bc7d-07cd-4203-8a25-7b16b549851b',
            '7622da34-51b6-4661-98ae-a57d40806008', 'bd456d8f-d36e-434a-8051-ff3997253802'])
    return eids


def criteria_opto_eids(eids, max_lapse=0.2, max_bias=0.3, min_trials=200, one=None):
    if one is None:
        one = ONE()
    use_eids = []
    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, laser_stimulation=True)
            lapse_l = 1 - (np.sum(trials.loc[trials['signed_contrast'] == -1, 'choice'] == 1)
                           / trials.loc[trials['signed_contrast'] == -1, 'choice'].shape[0])
            lapse_r = 1 - (np.sum(trials.loc[trials['signed_contrast'] == 1, 'choice'] == -1)
                           / trials.loc[trials['signed_contrast'] == 1, 'choice'].shape[0])
            bias = np.abs(0.5 - (np.sum(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)
                                 / np.shape(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)[0]))
            details = one.get_details(eid)
            if ((lapse_l < max_lapse) & (lapse_r < max_lapse) & (trials.shape[0] > min_trials)
                    & (bias < max_bias) & ('laser_stimulation' in trials.columns)):
                use_eids.append(eid)
            elif 'laser_stimulation' not in trials.columns:
                print('No laser_stimulation data for %s %s' % (details['subject'], details['start_time'][:10]))
            else:
                print('%s %s excluded (n_trials: %d, lapse_l: %.2f, lapse_r: %.2f, bias: %.2f)'
                      % (details['subject'], details['start_time'][:10], trials.shape[0], lapse_l, lapse_r, bias))
        except Exception:
            print('Could not load session %s' % eid)
    return use_eids

def load_exp_smoothing_trials(eids, one):
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
    for j, eid in enumerate(eids):
        try:
            # Load in trials vectors
            trials = load_trials(eid, invert_stimside=True, one=one)
            stimuli_arr.append(trials['signed_contrast'].values)
            actions_arr.append(trials['choice'].values)
            stim_sides_arr.append(trials['stim_side'].values)
            session_uuids.append(eid)
        except:
            print(f'Could not load trials for {eid}')

    # Get maximum number of trials across sessions
    max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

    # Pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials
    stimuli = np.array([np.concatenate((stimuli_arr[k], np.zeros(max_len-len(stimuli_arr[k]))))
                        for k in range(len(stimuli_arr))])
    actions = np.array([np.concatenate((actions_arr[k], np.zeros(max_len-len(actions_arr[k]))))
                        for k in range(len(actions_arr))])
    stim_side = np.array([np.concatenate((stim_sides_arr[k],
                                          np.zeros(max_len-len(stim_sides_arr[k]))))
                          for k in range(len(stim_sides_arr))])
    session_uuids = np.array(session_uuids)

    return actions, stimuli, stim_side, session_uuids


def fit_psychfunc(stim_levels, n_trials, proportion):
    # Fit a psychometric function with two lapse rates
    #
    # Returns vector pars with [bias, threshold, lapselow, lapsehigh]
    from ibl_pipeline.utils import psychofit as psy
    assert(stim_levels.shape == n_trials.shape == proportion.shape)
    if stim_levels.max() <= 1:
        stim_levels = stim_levels * 100

    pars, _ = psy.mle_fit_psycho(np.vstack((stim_levels, n_trials, proportion)),
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array([0, 20, 0.05, 0.05]),
                                 parmin=np.array([-100, 5, 0, 0]),
                                 parmax=np.array([100, 100, 1, 1]))
    return pars


def plot_psychometric(trials, ax, **kwargs):
    from ibl_pipeline.utils import psychofit as psy
    if trials['signed_contrast'].max() <= 1:
        trials['signed_contrast'] = trials['signed_contrast'] * 100

    stim_levels = np.sort(trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, trials.groupby('signed_contrast').size(),
                         trials.groupby('signed_contrast').mean()['right_choice'])

    # plot psychfunc
    sns.lineplot(x=np.arange(-27, 27), y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)),
                 ax=ax, **kwargs)

    # plot psychfunc: -100, +100
    sns.lineplot(x=np.arange(-36, -31), y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
                 ax=ax, **kwargs)
    sns.lineplot(x=np.arange(31, 36), y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
                 ax=ax, **kwargs)

    # now break the x-axis
    trials['signed_contrast'].replace(-100, -35)
    trials['signed_contrast'].replace(100, 35)

    # plot datapoints with errorbars on top
    sns.lineplot(x=trials['signed_contrast'], y=trials['right_choice'], ax=ax,
                     **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'ci':68}, **kwargs})

    ax.set(xticks=[-35, -25, -12.5, 0, 12.5, 25, 35], xlim=[-40, 40], ylim=[0, 1.02],
           yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '25', '50', '75', '100'],
           ylabel='Right choices', xlabel='Contrast (%)')
    ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                       size='small')
    break_xaxis()


def break_xaxis(y=-0.004, **kwargs):

    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')


def px_to_mm(dlc_df, camera='left', width_mm=66, height_mm=54):
    """
    Transform pixel values to millimeter

    Parameters
    ----------
    width_mm:  the width of the video feed in mm
    height_mm: the height of the video feed in mm
    """

    # Set pixel dimensions for different cameras
    if camera == 'left':
        px_dim = [1280, 1024]
    elif camera == 'right' or camera == 'body':
        px_dim = [640, 512]

    # Transform pixels into mm
    for key in list(dlc_df.keys()):
        if key[-1] == 'x':
            dlc_df[key] = dlc_df[key] * (width_mm / px_dim[0])
        if key[-1] == 'y':
            dlc_df[key] = dlc_df[key] * (height_mm / px_dim[1])
    dlc_df['units'] = 'mm'

    return dlc_df


def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv = np.sum(u*v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv])/2.0
    uc, vc = np.linalg.solve(A, B)
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1 = np.mean(Ri_1)
    return xc_1, yc_1, R_1


def pupil_features(dlc_df):
    vec_x = [dlc_df['pupil_left_r_x'], dlc_df['pupil_right_r_x'],
             dlc_df['pupil_top_r_x']]
    vec_y = [dlc_df['pupil_left_r_y'], dlc_df['pupil_right_r_y'],
             dlc_df['pupil_top_r_y']]
    x = np.zeros(len(vec_x[0]))
    y = np.zeros(len(vec_x[0]))
    diameter = np.zeros(len(vec_x[0]))
    for i in range(len(vec_x[0])):
        try:
            x[i], y[i], R = fit_circle([vec_x[0][i], vec_x[1][i], vec_x[2][i]],
                                       [vec_y[0][i], vec_y[1][i], vec_y[2][i]])
            diameter[i] = R*2
        except:
            x[i] = np.nan
            y[i] = np.nan
            diameter[i] = np.nan
    return x, y, diameter


def butter_filter(signal, highpass_freq=None, lowpass_freq=None, order=4, fs=2500):

    # The filter type is determined according to the values of cut-off frequencies
    Fn = fs / 2.
    if lowpass_freq and highpass_freq:
        if highpass_freq < lowpass_freq:
            Wn = (highpass_freq / Fn, lowpass_freq / Fn)
            btype = 'bandpass'
        else:
            Wn = (lowpass_freq / Fn, highpass_freq / Fn)
            btype = 'bandstop'
    elif lowpass_freq:
        Wn = lowpass_freq / Fn
        btype = 'lowpass'
    elif highpass_freq:
        Wn = highpass_freq / Fn
        btype = 'highpass'
    else:
        raise ValueError("Either highpass_freq or lowpass_freq must be given")

    # Filter signal
    b, a = butter(order, Wn, btype=btype, output='ba')
    if len(signal.shape) > 1:
        filtered_data = filtfilt(b=b, a=a, x=signal, axis=1)
    else:
        filtered_data = filtfilt(b=b, a=a, x=signal)

    return filtered_data

