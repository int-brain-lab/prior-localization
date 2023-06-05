# this a special pipeline to


# Third party libraries

# IBL libraries
# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path

# Third party libraries
import pandas as pd
# from dask.distributed import Client
# from dask_jobqueue import SLURMCluster

# IBL libraries
from one.api import ONE
from brainwidemap import bwm_query
from prior_localization.params import CACHE_PATH
CACHE_PATH.mkdir(parents=True, exist_ok=True)

# motor signal extraction libraries
import sys
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
import numpy as np
import math
import brainbox.behavior.wheel as wh
from brainbox.processing import bincount2D
from brainbox.io.one import SessionLoader

_logger = logging.getLogger('prior_pipelines')


one = ONE()
ba = AllenAtlas()
br = BrainRegions()

T_BIN = 0.02 # this is an important parameter


Fs = {'left':60,'right':150, 'body':30}

# specify binning type, either bins or sampling rate; 

sr = {'licking':'T_BIN','whisking_l':'T_BIN', 'whisking_r':'T_BIN', 
      'wheeling':'T_BIN','nose_pos':'T_BIN', 'paw_pos_r':'T_BIN', 
      'paw_pos_l':'T_BIN'}

blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]
cdi = {0.8:blue_left,0.2:red_right,0.5:'g',-1:'cyan',1:'orange'}



def get_all_sess_with_ME():
    # get all bwm sessions with dlc
    all_sess = one.alyx.rest('sessions', 'list', 
                              project='ibl_neuropixel_brainwide_01',
                              task_protocol="ephys", 
                              dataset_types='camera.ROIMotionEnergy')
    eids = [s['url'].split('/')[-1] for s in all_sess]
    
    return eids  

motor_eids = get_all_sess_with_ME()


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def get_licks(XYs):
    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''  
    
    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
           thr = np.nanstd(np.diff(c))/4
           licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
    return sorted(list(set.union(*licks))) 


def get_dlc_XYs(eid, video_type, query_type='remote'):
    #video_type = 'left'    
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy',
                             query_type=query_type) 
    cam = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.dlc.pqt', 
                           query_type=query_type)
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    
    return Times, XYs   

def cut_behavior(one, eid, duration =0.4, lag = -0.6,
                 align='stimOn_times', stim_to_stim=False, 
                 endTrial=False, query_type='remote',pawex=False):
    '''get_dlc_XYsdlc
    cut segments of behavioral time series for PSTHs
    
    param: eid: session eid
    param: align: in stimOn_times, firstMovement_times, feedback_times    
    param: lag: time in sec wrt to align time to start segment
    param: duration: length of cut segment in sec 
    '''
    # get wheel speed    
    wheel = one.load_object(eid, 'wheel', query_type=query_type)
    pos, times_w = wh.interpolate_position(wheel.timestamps,
                                           wheel.position, freq=1/T_BIN)
    v = np.append(np.diff(pos),np.diff(pos)[-1]) 
    v = abs(v) 
    v = v/max(v)  # else the units are very small

    sl = SessionLoader(one, eid)
    sl.load_motion_energy(views=['left', 'right'])
    sl.load_pose(views=['left', 'right'], likelihood_thr=0.9)

    # load whisker motion energy, separate for both cams
    times_me_l, whisking_l0 = sl.motion_energy['leftCamera']['times'], sl.motion_energy['leftCamera']['whiskerMotionEnergy']
    times_me_r, whisking_r0 = sl.motion_energy['rightCamera']['times'], sl.motion_energy['rightCamera']['whiskerMotionEnergy']

    points = np.unique(['_'.join(x.split('_')[:-1]) for x in sl.pose['leftCamera'].columns])
    times_l, XYs_l = sl.pose['leftCamera']['times'], {point: np.array([sl.pose['leftCamera'][f'{point}_x'],
                                                                       sl.pose['leftCamera'][f'{point}_y']])
                                                      for point in points if point != ''}
    times_r, XYs_r = sl.pose['rightCamera']['times'], {point: np.array([sl.pose['rightCamera'][f'{point}_x'],
                                                                       sl.pose['rightCamera'][f'{point}_y']])
                                                       for point in points if point != ''}

    DLC = {'left': [times_l, XYs_l], 'right': [times_r, XYs_r]}
    
    # get licks using both cameras    
    lick_times = []
    for video_type in ['right','left']:
        times, XYs = DLC[video_type]
        r = get_licks(XYs)
        try :
            idx = np.where(np.array(r)<len(times))[0][-1]  # ERROR HERE ...    
            lick_times.append(times[r[:idx]])
        except :
            print('ohoh')
    
    lick_times = sorted(np.concatenate(lick_times))
    R, times_lick, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    lcs = R[0]    
    # get paw position, for each cam separate
    
    if pawex:
        paw_pos_r0 = XYs_r['paw_r']
        paw_pos_l0 = XYs_l['paw_r']    
    else:
        paw_pos_r0 = (XYs_r['paw_r'][0]**2 + XYs_r['paw_r'][1]**2)**0.5
        paw_pos_l0 = (XYs_l['paw_r'][0]**2 + XYs_l['paw_r'][1]**2)**0.5
    
    # get nose x position from left cam only
    nose_pos0 = XYs_l['nose_tip'][0]
    licking = []
    whisking_l = []
    whisking_r = []
    wheeling = [] 
    nose_pos = []
    paw_pos_r = []
    paw_pos_l = []
        
    DD = []
           
    pleft = []
    sides = []
    choices = []
    T = [] 
    difs = [] # difference between stim on and last wheel movement
    d = (licking, whisking_l, whisking_r, wheeling,
         nose_pos, paw_pos_r, paw_pos_l,
         pleft, sides, choices, T, difs)
    ds = ('licking','whisking_l', 'whisking_r', 'wheeling',
         'nose_pos', 'paw_pos_r', 'paw_pos_l',
         'pleft', 'sides', 'choices', 'T', 'difs')
         
    D = dict(zip(ds,d))
    
    # continuous time series of behavior and stamps
    behaves = {'licking':[times_lick, lcs],
               'whisking_l':[times_me_l, whisking_l0], 
               'whisking_r':[times_me_r, whisking_r0], 
               'wheeling':[times_w, v],
               'nose_pos':[times_l, nose_pos0],
               'paw_pos_r':[times_r,paw_pos_r0], 
               'paw_pos_l':[times_l,paw_pos_l0]}
    trials = one.load_object(eid, 'trials', query_type=query_type)    
    wheelMoves = one.load_object(eid, 'wheelMoves', query_type=query_type)
    
    print('cutting data')
    trials = one.load_object(eid, 'trials', query_type=query_type)
    evts = ['stimOn_times', 'feedback_times', 'probabilityLeft',
            'choice', 'feedbackType','firstMovement_times']
            
    kk = 0     
    for tr in range(len(trials['intervals'])): 
        
        a = wheelMoves['intervals'][:,1]

        if stim_to_stim:
            start_t = trials['stimOn_times'][tr]    
                    
        elif align == 'wheel_stop':            
            start_t = a + lag    
            
        else:                                
            start_t = trials[align][tr] + lag     
                
        if np.isnan(trials['contrastLeft'][tr]):
            cont = trials['contrastRight'][tr]            
            side = 0  # right side stimulus
        else:   
            cont = trials['contrastLeft'][tr]         
            side = 1  # left side stimulus                   
                  
        sides.append(side) 
        
        if endTrial:
            choices.append(trials['choice'][tr+1])
        else:                              
            choices.append(trials['choice'][tr])   
                 
        pleft.append(trials['probabilityLeft'][tr])
         
        for be in behaves: 
            times = behaves[be][0] 
            series = behaves[be][1] 
            start_idx = find_nearest(times,start_t)        
            if stim_to_stim:  
                end_idx = find_nearest(times, trials['stimOn_times'][tr + 1])
            else:
                if sr[be] == 'T_BIN':
                    end_idx = start_idx + int(duration/T_BIN)
                else:
                    fs = sr[be]
                    end_idx = start_idx + int(duration*fs)              
            
            if (pawex and ('paw' in be)): #for illustration on frame
                D[be].append([series[0][start_idx:end_idx],
                              series[1][start_idx:end_idx]])            
            else:  
                 # bug inducing 
                if start_idx > len(series):
                    print('start_idx > len(series)')
                    break      
                D[be].append(series[start_idx:end_idx])         
                  
        T.append(tr)
        kk+=1

    print(kk, 'trials used')
    return(D)


def load_motor(one, eid):

    assert eid in motor_eids, "no motor signals for this session"
    motor_regressors = cut_behavior(one, eid,duration =2, lag = -1,query_type='auto') # very large interval
    motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']
    regressors = dict(filter(lambda i:i[0] in motor_signals_of_interest, motor_regressors.items()))
    return regressors


def cache_motor(subject, eid, regressors):
    """
    Take outputs of load_motor) and cache them to disk in the folder defined in the params.py
    file in this repository, using a nested subject -> session folder structure.

    doesn't check id the files already exist...

    Returns the metadata filename and regressors filename.
    """
    sesspath = Path(CACHE_PATH).joinpath('motor').joinpath(subject).joinpath(eid)
    sesspath.mkdir(parents=True, exist_ok=True)
    curr_t = dt.now()
    fnbase = str(curr_t.date())
    metadata_fn = sesspath.joinpath(fnbase + '_motor_metadata.pkl')
    data_fn = sesspath.joinpath(fnbase + '_motor_regressors.pkl')
    metadata = {
        'subject': subject,
        'eid': eid
    }
    with open(metadata_fn, 'wb') as fw:
        pickle.dump(metadata, fw)
    with open(data_fn, 'wb') as fw:
        pickle.dump(regressors, fw)

    del regressors
    return metadata_fn, data_fn


def delayed_load(eid):
    try:
        return load_motor(eid)
    except KeyError:
        pass

def delayed_save(subject, eid, outputs):
    return cache_motor(subject, eid, outputs)


if __name__ == '__main__':
    # Parameters
    ALGN_RESOLVED = True
    DATE = str(dt.today())
    MAX_LEN = None
    T_BEF = 0.6
    T_AFT = 0.6
    BINWIDTH = 0.02
    ABSWHEEL = False
    WHEEL = False
    QC = True
    TYPE = 'primaries'
    MERGE_PROBES = True
    # End parameters

    # Construct params dict from above
    params = {
        'max_len': MAX_LEN,
        't_before': T_BEF,
        't_after': T_AFT,
        'binwidth': BINWIDTH,
        'abswheel': ABSWHEEL,
        'ret_qc': QC,
        'wheel': WHEEL,
    }


    bwm_df = bwm_query(freeze="2022_10_initial").set_index(["subject", "eid"]) # frozen dataset

    dataset_futures = []

    for eid in bwm_df.index.unique(level='eid'):

        session_df = bwm_df.xs(eid, level='eid')
        subject = session_df.index[0]
        load_outputs = delayed_load(eid)
        save_future = delayed_save(subject, eid, load_outputs)
        dataset_futures.append([subject, eid, save_future])


    N_CORES = 4

    cluster = SLURMCluster(cores=N_CORES,
                        memory='32GB',
                        processes=1,
                        queue="shared-cpu",
                        walltime="01:15:00",
                        log_directory='/srv/beegfs/scratch/users/h/hubertf/dask-worker-logs',
                        interface='ib0',
                        extra=["--lifetime", "60m", "--lifetime-stagger", "10m"],
                        job_cpu=N_CORES,
                        env_extra=[
                            f'export OMP_NUM_THREADS={N_CORES}',
                            f'export MKL_NUM_THREADS={N_CORES}',
                            f'export OPENBLAS_NUM_THREADS={N_CORES}'
                        ])

    # cluster = LocalCluster()
    cluster.scale(20)

    client = Client(cluster)

    tmp_futures = [client.compute(future[2]) for future in dataset_futures]


    # Run below code AFTER futures have finished!
    dataset = [{
        'subject': x[0],
        'eid': x[1],
        'meta_file': x[-1][0],
        'reg_file': x[-1][1]
    } for i, x in enumerate(dataset_futures)]
    dataset = pd.DataFrame(dataset)

    outdict = {'params': params, 'dataset_filenames': dataset}
    with open(Path(CACHE_PATH).joinpath(DATE + '_motor_metadata.pkl'), 'wb') as fw:
        pickle.dump(outdict, fw)

