#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:29:32 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap
from plot_utils import comb_regs_df, get_within_region_mean_var, get_predprob_vals
from yanliang_brain_slice_plot import get_cmap
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units
sns.set(font_scale=1.5)
sns.set_style('ticks')

# get reference cluster dataframe
julias_clusters = bwm_units(ONE(base_url='https://openalyx.internationalbrainlab.org',
                                password='international'))
julias_clusters['sessreg'] = julias_clusters.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)
ref_clusters = julias_clusters[['uuids','sessreg']]

CUSTOM_SESSREG_FILTER = None # can use something other than ref_clusters
                             # if this is a tuple (min_units, min_reg)

'''
01-04-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv
01-04-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv
01-04-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv
02-04-2023_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv
01-04-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv
'''

DATE = '04-05-2023'
VARI = 'block'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '04-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'
FIG_SUF = '.png'

FOCUS_REGIONS = []

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# filter results
if not (CUSTOM_SESSREG_FILTER is None):
    min_units, min_reg = CUSTOM_SESSREG_FILTER
    res_table = pd.read_csv(file_all_results)
    res_table = res_table.loc[res_table['n_units']>=min_units]
    res_table = res_table.loc[res_table['region']!='void']
    res_table = res_table.loc[res_table['region']!='root']
    reg_counts = res_table['region'].value_counts()
    res_table = res_table.loc[res_table['region'].isin(reg_counts[reg_counts>=min_reg].index)]
    
    xy_table = pd.read_pickle(file_xy_results)
    eid_regs_filtered = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
    xy_table = xy_table.loc[xy_table['eid_region'].isin(eid_regs_filtered)]
    
    # check clusters
    cuuids = np.concatenate(list(xy_table['cluster_uuids']))
    assert set(ref_clusters['uuids']).issubset(set(cuuids))
else:
    #filter according to reference  session_regions
    res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
    res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]
    xy_table = xy_table.loc[xy_table['eid_region'].isin(ref_clusters['sessreg'])]

    # check clusters
    cuuids = np.concatenate(list(xy_table['cluster_uuids']))
    assert set(cuuids) == set(ref_clusters['uuids'])

# combine regions and save
save_comb_regs_data = comb_regs_df(res_table, 
                                   USE_ALL_BERYL_REGIONS=True)
regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)
n_sig = regs_table['combined_sig'].sum()
f_sig = regs_table['combined_sig'].mean()
wi_means, wi_vars = get_within_region_mean_var(res_table)
save_comb_regs_data.to_csv(
    f'decoding_processing/{DATE}_{VARI}_regs_nsig{n_sig}_fsig{f_sig:.3f}_wi2ovar{np.mean(wi_vars)/np.var(wi_means):.3f}.csv')

# get weights and save
ws = np.concatenate(list(xy_table['weights']), axis=-1)[:, :, 0, :]
ws = ws.reshape((50, -1))
ws_dict = {f'ws_fold{i%5}_runid{i//5}': ws[i, :] for i in range(50)}
save_cluster_weights = pd.DataFrame({'cluster_uuids': cuuids,
                                     **ws_dict})
save_cluster_weights.to_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')

# get cluster, region, session lists and save
pd.DataFrame({'cluster_uuids': cuuids}).to_csv(f'decoding_processing/clusters_regions_sessions/{DATE}_{VARI}_clusters.csv')
pd.DataFrame({'regions': np.unique(res_table['region'])}).to_csv(f'decoding_processing/clusters_regions_sessions/{DATE}_{VARI}_regions.csv')
pd.DataFrame({'session_eids': np.unique(res_table['eid'])}).to_csv(f'decoding_processing/clusters_regions_sessions/{DATE}_{VARI}_sessions.csv')

# %%

regs = np.array(regs_table['region'])
fs_regs = np.array(regs_table['frac_sig'])
assert not np.any(np.isnan(fs_regs))

brain_SwansonFlat_results(regs,
                          fs_regs,
                          filename=f'{VARI}_swanson_fs' + FIG_SUF,
                          cmap=get_cmap(VARI),
                          clevels=[0, 0.55],
                          ticks=None,
                          extend='max',
                          cbar_orientation='horizontal',
                          value_title='Fraction of significant sessions')

# generate_sag_slices(regs, fs_regs)

ms_regs = np.array(regs_table['values_median_sig'])

brain_SwansonFlat_results(regs[~np.isnan(ms_regs)],
                          ms_regs[~np.isnan(ms_regs)],
                          filename=f'{VARI}_swanson_ms' + FIG_SUF,
                          cmap=get_cmap(VARI),
                          clevels=[0.5, 1.0],
                          ticks=None,
                          extend=None,
                          cbar_orientation='horizontal',
                          value_title='Median significant balanced accuracy')

n_regs = np.array(regs_table['n_sessions'])
assert not np.any(n_regs == 0)
n_regs = np.log(n_regs) / np.log(2)

brain_SwansonFlat_results(regs,
                          n_regs,
                          filename=f'{VARI}_swanson_n' + FIG_SUF,
                          cmap=get_cmap(VARI),
                          clevels=[None, None],
                          ticks=([1, 2, 3, 4, 5], [2, 4, 8, 16, 32]),
                          extend=None,
                          cbar_orientation='vertical',
                          value_title='N Sessions')

# assert regions have a fisher combined p-value<0.05,
#        sorted by best median performance (TOPN values plotted),
#        and greater median performance than the median of the null

regions = np.array(regs_table.loc[regs_table['combined_sig_corr'], 'region'])


def get_vals(reg):
    return np.array(res_table.loc[res_table["region"] == reg, "score"])


values = np.array([get_vals(reg) for reg in regions])


def get_pvals(reg):
    return np.array(res_table.loc[res_table["region"] == reg, "p-value"])


values_sig = np.array([(get_pvals(reg) < 0.05) + 0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(
    regs_table.loc[regs_table['combined_sig_corr'], 'null_median_of_medians'])
acr_plotted = bar_results(regions,
                          values,
                          comb_vals,
                          comb_nulls,
                          fillcircle_eids_unordered=values_sig,
                          filename=f'{VARI}_bars' + FIG_SUF,
                          YMIN=np.min([np.min(v) for v in values]),
                          ylab='Bal. Acc.',
                          ticks=([0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8]),
                          #TOP_N=15,
                          sort_args=None,
                          bolded_regions=FOCUS_REGIONS)
# check criteria.
for reg in acr_plotted:
    print(reg)
    assert np.median(get_vals(reg)) > np.median(
        res_table.loc[res_table['region'] == reg, 'median-null'])

# %% plot single session traces
sns.set_style('ticks')

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

# load single trial data
eid = '1191f865-b10a-45c8-9c48-24a980fd9402'
region = 'ORBvl'
eid = 'dd4da095-4a99-4bf3-9727-f735077dba66'
region = 'PL'
eid = 'b658bc7d-07cd-4203-8a25-7b16b549851b'
region = 'CP'
xy_vals = get_xy_vals(xy_table, eid, region)
er_vals = get_res_vals(res_table, eid, region)

l = xy_vals['regressors'].shape[0]
X = np.squeeze(xy_vals['regressors']).T
ws = np.squeeze(xy_vals['weights'])
assert len(ws.shape) == 3
W = np.stack([np.ndarray.flatten(ws[:, :, i]) for i in range(ws.shape[2])]).T
assert W.shape[0] == 50
mask = xy_vals['mask']
preds = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
predprobs = get_predprob_vals(xy_table, eid, region)
targs = np.squeeze(xy_vals['targets'])
trials = np.arange(len(mask))[[m == 1 for m in mask]]

plt.figure(figsize=(14, 6))

plt.title(
    f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
plt.plot(trials, 1-targs, '-', c='k', lw=4)
plt.plot(trials, 1-predprobs, '-', c='mediumpurple')
plt.yticks([0, .5, 1])
plt.ylim(-0.1, 1.1)
plt.xlim(0, len(mask))
plt.legend(['Right Biased Block',
            'Probability of right prediction \n(across 10 models)'],
           frameon=True,
           loc=(0.9, 1.1))
plt.xlabel('Trials')
plt.ylabel('Average predicted \nright block')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/{VARI}_trace.png', dpi=200)
plt.show()

plt.figure(figsize=(14.2, 6))
plt.title(
    f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")

nscores = []
for i in range(3,6):
    null_example = pd.read_pickle(f'decoding_results/null_example/04-05-2023_CP_target_pLeft_timeWindow_-0_4_-0_1_pseudo_id_{i}__binsize=300.0_lags=None_mergedProbes_True.pkl')
    ntargs = np.array(null_example['fit'][0]['target'])[:,0]
    plt.plot(trials, ntargs, lw=4)
    nscores.append(np.mean([null_example['fit'][i]['balanced_acc_test_full'] for i in range(10)]))
plt.legend([f'pseudo-session bal. acc.: {ns:.3f}'for ns in nscores],
           frameon=True,
           loc=(0.9, 1.1))   
plt.yticks([0, .5, 1])
plt.ylim(-0.1, 1.1)
plt.xlim(0, len(mask))
plt.xlabel('Trials')
plt.ylabel(' \n Right block')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/{VARI}_trace_nullexamples.png', dpi=200)
plt.show()

plt.figure(figsize=(5, 4))
plt.title(
    f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
# plt.plot(trials, targs, '-', c='k', lw=4)
# plt.plot(trials, predprobs, '-', c='mediumpurple')
# plt.legend(['Left Biased Block',
#             'Probability of left prediction \n(across 10 models)'],
#            frameon=True,
#            loc=(0.9, 1.1))
plt.plot(trials[targs==0],
         1-predprobs[targs==0],
         'o', c = (255/255, 48/255, 23/255),
         lw=2,ms=4)
plt.plot(trials[targs==1], 
         1-predprobs[targs==1],
         'o', c = (34/255,77/255,169/255),
         lw=2,ms=4)
plt.legend(['Prediction given choice$=$R',
            'Prediction given choice$=$L'],
           frameon=True,
           loc=(0.9,1.1))
plt.yticks([0, .5, 1])
plt.ylim(-0.1, 1.1)
plt.xlim(100, 400)
plt.xlabel('Trials')
plt.ylabel('Average predicted \nright block')
plt.tight_layout()
plt.savefig(f'decoding_figures/{VARI}_trace.svg', dpi=200)
plt.show()

plt.figure(figsize=(6,4))
nscores = []
for i in range(1,1001):
    null_example = pd.read_pickle(f'decoding_results/null_example/04-05-2023_CP_target_pLeft_timeWindow_-0_4_-0_1_pseudo_id_{i}__binsize=300.0_lags=None_mergedProbes_True.pkl')
    ntargs = np.array(null_example['fit'][0]['target'])[:,0]
    nscores.append(np.mean([null_example['fit'][i]['balanced_acc_test_full'] for i in range(10)]))
plt.hist(nscores, 
         bins=32, 
         histtype='step',
         density=True,
         lw=4, color='k', alpha=0.5)
median_val = np.median(nscores)
real_val = er_vals['score']
plt.plot(median_val*np.ones(10), np.linspace(0,15,10), 
         'k-', lw=4)
plt.plot(real_val*np.ones(10), np.linspace(0,15,10), 
         '-', c='mediumpurple', lw=4)

plt.xlim(0.45,1.0)
plt.xlabel('Balanced accuracy')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/{VARI}_nulldist_defineeffectsize.png', dpi=200)
plt.show()
# %%

regions = np.unique(res_table['region'])
regions = np.array([reg for reg in regions if not (
    (reg == 'root') or (reg == 'void'))])
regions = regions[np.argwhere(regions == 'SPVI')[0][0]:]
# regions = np.array(['MRN', 'CP', 'SUB'])
regions = np.array(['CA3'])
for my_reg in regions:
    xy_eids = [er.split('_')[0] for er in xy_table['eid_region']
               if er.split('_')[1] == my_reg]
    # assert (len(xy_rgrs) == len(xy_trgs)) and (len(xy_rgrs)==len(xy_prds))
    # assert len(xy_rgrs) == len(xy_eids)
    Ns = [get_res_vals(res_table, xy_eids[i], my_reg)['n_units']
          for i in range(len(xy_eids))]
    N = np.max(Ns)
    pvals = [get_res_vals(res_table, xy_eids[i], my_reg)['p-value']
             for i in range(len(xy_eids))]
    sargs = np.argsort(pvals)
    xy_eids = np.array(xy_eids)
    xy_eids = xy_eids[sargs]

    Nsubplots = 2 * len(xy_eids) + 1
    fig, axs = plt.subplots(Nsubplots,
                            figsize=(40, 3 * Nsubplots))  # int(0.6*(N+1))

    axs[0].set_title(f'{acronym2name(my_reg)} ({my_reg})',
                     fontsize=30)
    axs[0].hist(pvals)
    axs[0].set_ylabel('Count')
    axs[0].set_xlabel('P-values across sessions')
    axs[0].set_xlim(0, 1)

    for ei in range(len(xy_eids)):
        xyi = ei * 2 + 1
        my_eid = xy_eids[ei]
        xy_vals = get_xy_vals(xy_table, my_eid, my_reg)

        xy_rgr = np.squeeze(xy_vals['regressors']).T
        ws = np.squeeze(xy_vals['weights'])
        assert len(ws.shape) == 3
        xy_w = np.stack([np.ndarray.flatten(ws[:, :, i])
                        for i in range(ws.shape[2])]).T
        assert xy_w.shape[0] == 50
        xy_w = np.mean(xy_rgr, axis=1) * xy_w  # if weight by mean activity
        xy_prd = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
        xy_trg = np.squeeze(xy_vals['targets'])
        xy_prm = xy_vals['params']
        # print(xy_rgr.shape)

        '''
        --------------------------------------------------
        first plot: violin plots of activity and predictions
        --------------------------------------------------
        '''
        MAX_SPIKES = np.max(xy_rgr)

        x = []
        xy_n = xy_rgr.shape[0]
        pr_xval = xy_n
        for i in range(xy_rgr.shape[0]):
            for t in range(xy_rgr.shape[-1]):
                x.append([i, t, xy_rgr[i, t], xy_trg[t]])

        for t in range(xy_rgr.shape[1]):
            x.append([pr_xval, t, MAX_SPIKES * xy_prd[t], xy_trg[t]])

        df = pd.DataFrame(x, columns=['neuron',
                                      'trial',
                                      'spikes',
                                      'target'])
        mr = res_table.loc[(res_table['eid'] == my_eid) &
                           (res_table['region'] == my_reg)]
        assert len(mr) == 1
        mr = mr.iloc[0]
        nt = len(xy_trg)
        nt0 = len(xy_trg[xy_trg == 0])
        nt1 = len(xy_trg[xy_trg == 1])
        assert nt == (nt0 + nt1)
        axs[xyi].set_title(
            f'eid:{my_eid}, score:{mr["score"]:.3f}, p:{mr["p-value"]:.3f}, frac_w:{mr["frac_large_w"]:.3f}, gini_w:{mr["gini_w"]:.3f}, n_trials:{nt}, n_trials0:{nt0}, n_trials1:{nt1}',
            fontsize=30)
        sns.violinplot(ax=axs[xyi], data=df,
                       x="neuron", y="spikes",
                       hue="target", split=True,
                       cut=0, linewidth=0)
        lxs = np.linspace(-0.4, 0.4)
        lys = MAX_SPIKES * np.ones_like(lxs)
        axs[xyi].plot(lxs + pr_xval, lys, 'r', lw=4)
        axs[xyi].plot(lxs + pr_xval, np.zeros_like(lxs), 'r', lw=4)
        axs[xyi].text(pr_xval + 0.5, MAX_SPIKES * (1), 'Y*=1')
        axs[xyi].text(pr_xval + 0.5, MAX_SPIKES * (0), 'Y*=0')
        axs[xyi].text(pr_xval + 0.6, MAX_SPIKES * (0.5),
                      'Prediction of logistic decoder, Y*, \nbetween 0 and 1')
        tlabels = np.concatenate((np.arange(xy_n), np.array([pr_xval])))
        newlabels = ['Y*' if l == pr_xval else str(l) for l in tlabels]
        axs[xyi].set_xticks(tlabels)
        axs[xyi].set_xticklabels(newlabels,
                                 rotation=-45)
        # align weights to violin plot
        # define heatmap edges relative to violin plot x-values
        # as a fraction of total violin plot x-axis (c_0, c_f),
        # then solve for the x-limits to choose in violin plot (Svec)
        # so that neuron numbers {0, 1... N-1} line up
        c_0, c_f = 0, .80
        M_inv = np.linalg.inv(np.array([[1 - c_0, c_0], [1 - c_f, c_f]]))
        Svec = np.matmul(M_inv, np.array([[-0.5], [xy_n + 0.5]]))[:, 0]
        axs[xyi].set_xlim(Svec[0], Svec[1])
        axs[xyi].legend(loc='upper left', title='Target')

        '''
        --------------------------------------------
        second plot: distribution of decoder weights
        --------------------------------------------
        '''
        xy_w_abs = np.abs(xy_w)

        prms = np.squeeze(xy_prm[:, :, :, 1])
        assert len(prms.shape) == 2
        assert prms.shape[0] == 10
        assert prms.shape[1] == 5
        prms = np.reshape(np.array(np.ndarray.flatten(prms), dtype=float),
                          (50, 1))
        prms_plt = np.log10(prms)
        prms_plt = prms_plt - np.min(prms_plt)
        prms_plt = np.max(xy_w_abs) * prms_plt / \
            np.max(prms_plt) if np.max(prms_plt) > 0 else np.zeros_like(prms_plt)
        MW = np.hstack((xy_w_abs, prms_plt))

        w_ticks = np.arange(xy_n + 1)
        im, cbar = heatmap(MW, np.arange(50), w_ticks, ax=axs[xyi + 1],
                           cmap="YlGn", cbarlabel="Decoder weights (abs. value)",
                           aspect=xy_n / 900,
                           interpolation=None,
                           filternorm=False,
                           resample=False,
                           cbar_kw={'shrink': 0.7,
                                    'location': 'right'})
        w_ticklabels = ['' if w == xy_n else str(w) for w in w_ticks]
        axs[xyi + 1].set_xticks(w_ticks,
                                labels=w_ticklabels,
                                rotation=-45)
        prm_mini = np.argmin(prms)
        prm_min = np.min(prms)
        prm_maxi = np.argmax(prms)
        prm_max = np.max(prms)
        axs[xyi + 1].text(xy_n + 0.5, prm_mini + 1,
                          f'--- min: {prm_min:.1e}',
                          fontsize=8)
        axs[xyi + 1].text(xy_n + 0.5, prm_maxi + 1,
                          f'--- max: {prm_max:.1e}',
                          fontsize=8)
        axs[xyi + 1].text(xy_n - 0.25, 51,
                          'Decoder \nhyper-params \n(log spacing)',
                          va='top',
                          fontsize=8)
    # save_path = f'decoding_figures/block_bin_dist/{my_reg}.png'
    # axs = activity_and_decoding_weights(res_table, xy_table,
    #                                     my_reg, save_path)

    plt.tight_layout()
    plt.savefig(f'decoding_figures/{VARI}_bin_dist/{my_reg}.png', dpi=100)
    print(f'region complete: {my_reg}')
