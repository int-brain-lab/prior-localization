import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["figure.facecolor"] = 'white'
mpl.rcParams["axes.facecolor"] = 'white'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
import glob
import wfi_utils as wut
import scipy as sp
import scipy.ndimage

import glob
from sklearn.linear_model import Ridge

import torch
from tqdm import tqdm


# we load all of the weights
weights_all_timesteps = []
intercepts_all_time_steps = []
dates = [40,26,27,28,29,30,31,32,33,34,41]
timesteps = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
for i in range(11):

    print(i,dates[i],timesteps[i])
    
    # recovery for linux system
    temp_weights = pd.read_pickle('/home/users/h/hubertf/scratch/decoding/results/neural/widefield/'+str(dates[i])+'-06-2022_decode_pLeft_optBay_Ridge_align_stimOn_times_200_pseudosessions_allProbes_timeWindow_'+str(timesteps[i])+'_'+str(timesteps[i])+'_imposterSess_0_balancedWeight_0_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.w.pkl')
    temp_intercepts = pd.read_pickle('/home/users/h/hubertf/scratch/decoding/results/neural/widefield/'+str(dates[i])+'-06-2022_decode_pLeft_optBay_Ridge_align_stimOn_times_200_pseudosessions_allProbes_timeWindow_'+str(timesteps[i])+'_'+str(timesteps[i])+'_imposterSess_0_balancedWeight_0_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.i.pkl')

    weights_all_timesteps.append(temp_weights)
    intercepts_all_time_steps.append(temp_intercepts)

df_weights = pd.concat(weights_all_timesteps,keys=timesteps).reset_index()
df_weights['nb_weights'] = df_weights['weights'].apply(lambda x : x.shape[1]) 
df_weights.rename(columns={'level_0':'timestep'}, inplace=True)
df_intercepts = pd.concat(intercepts_all_time_steps,keys=timesteps).reset_index()
df_weights['intercepts'] = df_intercepts['intercepts']


subjects = glob.glob('/home/share/pouget_lab/wide_field_imaging/CSK-im-*') #we only have CSK-im-12 in local
eids = np.array([np.load(s + '/behavior.npy', allow_pickle=True).size for s in subjects])

options = {
    'align_time': 'stimOn_times',
    'wfi_hemispheres':['left','right']
}

frames_accuracies_0C = []
frames_pvalues_0C = []

for index in tqdm(range(eids.sum())): # eids.sum()

    eid_id = index % eids.sum()
    subj_id = np.sum(eid_id >= eids.cumsum())
    sess_id = eid_id - np.hstack((0, eids)).cumsum()[:-1][subj_id]

    sessiondf, wideFieldImaging_dict = wut.load_wfi_session(subjects[subj_id], sess_id)

    eid = sessiondf.eid[sessiondf.session_to_decode].unique()[0]
    print(eid)
    eid_sessiondf = sessiondf[sessiondf['eid'] == eid].reset_index(drop=True)
    wifi_activity = np.transpose(wideFieldImaging_dict['activity'],(1,2,0))

    side, stim, act, oracle_pLeft = wut.format_data(eid_sessiondf)
    prior = wut.optimal_Bayesian(act, side).numpy()

    reg_mask = wut.select_widefield_imaging_regions(wideFieldImaging_dict,np.unique(wideFieldImaging_dict['atlas']['acronym'].values),kwargs=options)

    accuracies_0C = np.zeros((len(timesteps),len(timesteps)))
    p_values_0C = np.zeros((len(timesteps),len(timesteps)))

    test_sess = df_weights[df_weights['session'] == eid]

    for i_decoder,t_decoder in enumerate(timesteps) :

        test_sess_t = test_sess[ test_sess['timestep']  == t_decoder].reset_index()

        for i_decoded,t in enumerate(timesteps) :

            options['wfi_nb_frames_start'] = t
            options['wfi_nb_frames_end'] = t

            preprocess_activity = wut.preprocess_widefield_imaging(wideFieldImaging_dict,reg_mask,options)

            # the score we want to compute is the accuracy on zero-contrast trial of the prediction => is it above chance level ?
            true_0c = (side[np.where((stim == 0) & (side != 0))] > 0) * 1 #[np.where((stim == 0) & (side != 0))]

            mean_acc_0c = []
            for i_fold in range(5):
                for i_run in range(10):
                    weights = test_sess_t['weights'][i_run][i_fold]
                    intercept = test_sess_t['intercepts'][i_run][i_fold]
                    clf = Ridge(alpha=1.0)
                    clf.coef_ = weights
                    clf.intercept_ = intercept

                    prediction = clf.predict(np.concatenate(preprocess_activity))
                    prediction_0c = (prediction[np.where((stim == 0) & (side != 0))]  > 0.5 )*1   #[np.where((stim == 0) & (side != 0))] 
                    accuracy_0c = np.sum( np.equal(prediction_0c,true_0c)) / true_0c.shape[0]
                    # print('decoder t:',t_decoder,'decoded t:',t,'R2',clf.score(np.concatenate(preprocess_activity),prior),'accuracy_0c',accuracy_0c)
                    mean_acc_0c.append(accuracy_0c)

            p = scipy.stats.ttest_1samp(mean_acc_0c,0.5).pvalue
            p_values_0C[i_decoder,i_decoded] = np.mean(p)
            accuracies_0C[i_decoder,i_decoded] = np.mean(mean_acc_0c)

    frames_accuracies_0C.append(accuracies_0C)
    frames_pvalues_0C.append(p_values_0C)


import pickle
with open("cross_decoding_results_zero_contrast.pkl","wb") as f:
    pickle.dump([frames_accuracies_0C,frames_pvalues_0C],f)


