#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:35:55 2023

@author: bensonb
"""
import os
import numpy as np
import pandas as pd
from plot_utils import comb_regs_df
from ibllib.atlas.plots import reorder_data
import matplotlib.pyplot as plt
import matplotlib
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


PATH_TO_ALLEN_COLOR_CSV = '../../allen_structure_tree.csv'
allen_color_data = np.genfromtxt(PATH_TO_ALLEN_COLOR_CSV, 
                                 delimiter=',',dtype=str)
def hex2rgba(hex):
    if '' == hex:
        return (0,0,0,0)
    while len(hex) < 6:
        hex = '0'+hex
    assert len(hex) == 6
    return matplotlib.colors.to_rgba('#'+hex)

reg2rgba_dict = {allen_color_data[i,3]:
                 hex2rgba(allen_color_data[i,13]) for i in range(1,allen_color_data.shape[0])}


def bar_results(acronyms_unordered, 
                values_eids_unordered, 
                values_unordered,
                nulls_unordered, 
                fillcircle_eids_unordered=None,
                filename='test.png', 
                ylab='',
                ticks=None,
                FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/braindelphi/decoding_figures/',
                YMIN=None,
                YMAX=None,
                TOP_N=np.nan,
                POOL_PROTOCOL='median',
                sort_args=None,
                bolded_regions=[]):
    '''
    

    Parameters
    ----------
    acronyms_unordered : array
        region acronyms
    values_eids_unordered : array of arrays
        each element in array is an array of all sessions' values
        first dimension corresponds to regions in acronyms_unordered
    values_unordered : array
        the combined values across eids in values_eids_unordered
        each element corresponds to regions in acronyms_unordered
    nulls_unordered : array
        the null value to plot as white circle on each region's bar.
        each element corresponds to regions in acronyms_unordered
    filename : str, optional
        The default is 'test.png'.
    ylab : str, optional
        The default is ''.
    ticks : tuple, (list of ticks, list of labels), y-ticks to use
    FILE_PATH : str, optional
        The default is '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'.
    YMIN : float, optional
        The default is None.
    TOP_N : int, optional
        only plot the top n values. The default is np.nan. sorted by maximum
        value conditioned on the center value being larger then the null center
    POOL_PROTOCOL : str, optional
        'median' or 'mean'.  how to do per-region pooling across sessions.
        The default is 'median'.
    sort_args : None or array
        if None and TOP_N is given, the top arguments are show after sorting
        by pooled values. (if pooled values are less than the null, 
                           then they are automatically considered the worst 
                           in sorting)
        if array and TOP_N is given, the first TOP_N arguments in sort_args
        will be plotted
    bolded_regions : None or array
        if not None, array of strings for each region which should be bolded
        in the x-axis
    Returns
    -------
    acronyms of TOP_N values

    '''
    # if POOL_PROTOCOL == 'median':
    #     values_unordered = np.array([np.median(vs) for vs in values_eids_unordered])
    # elif POOL_PROTOCOL == 'mean':
    #     values_unordered = np.array([np.mean(vs) for vs in values_eids_unordered])
    # else:
    #     raise ValueError('This value of POOL_PROTOCOL is not implemented.')
    # values_unordered
    if fillcircle_eids_unordered is None:
        fillcircle_eids_unordered = np.array([np.ones(len(vs)) for vs in values_eids_unordered])
    
    if not np.isnan(TOP_N):
        if sort_args is None:
            v_to_sort_on = values_unordered*np.maximum(values_unordered-nulls_unordered,0)
            sinds = np.argsort(v_to_sort_on)[::-1][:TOP_N]
        else:
            sinds = sort_args[:TOP_N]
        
        acronyms_unordered = acronyms_unordered[sinds]
        if len(nulls_unordered.shape)==1:
            nulls_unordered = nulls_unordered[sinds]
        elif len(nulls_unordered.shape)==2:
            nulls_unordered = np.vstack((nulls_unordered[0,:][sinds],
                                         nulls_unordered[1,:][sinds],
                                         nulls_unordered[2,:][sinds]))
        values_eids_unordered = values_eids_unordered[sinds]
        fillcircle_eids_unordered = fillcircle_eids_unordered[sinds]
        values_unordered = values_unordered[sinds]
    
    acronyms, values = reorder_data(acronyms_unordered, values_unordered)
    if len(nulls_unordered.shape)==1:
        print(acronyms_unordered,nulls_unordered)
        acronyms_tmp, nulls = reorder_data(acronyms_unordered, nulls_unordered)
        assert np.all(acronyms == acronyms_tmp)
        print(values,nulls)
        assert len(values) == len(nulls)
        nulls_errbars = None
    elif len(nulls_unordered.shape)==2:
        acronyms_tmpl, nulls_l = reorder_data(acronyms_unordered, nulls_unordered[0,:])
        acronyms_tmpm, nulls_m = reorder_data(acronyms_unordered, nulls_unordered[1,:])
        acronyms_tmph, nulls_h = reorder_data(acronyms_unordered, nulls_unordered[2,:])
        assert np.all(acronyms == acronyms_tmpl)
        assert np.all(acronyms == acronyms_tmpm)
        assert np.all(acronyms == acronyms_tmph)
        assert len(values) == len(nulls_l)
        assert len(values) == len(nulls_m)
        assert len(values) == len(nulls_h)
        nulls = nulls_m
        nulls_errbars = np.vstack((nulls_m-nulls_l, nulls_h-nulls_m))
        
        
    PLOT_TITLE = ''
    SAVE_PATH = os.path.join(FILE_PATH, filename)
    fig = plt.figure(figsize=(2.5 + int(len(acronyms)/4),5))
    plt.title(PLOT_TITLE)
    inds = np.arange(len(acronyms))
    plt.bar(inds, values, 
            color=[reg2rgba_dict[r] for r in acronyms],
            #edgecolor='k',
            #linewidth=1,
            )
    plt.errorbar(inds, nulls, yerr=nulls_errbars, 
                         ecolor='k', elinewidth=3, linestyle='')
    plt.plot(inds, nulls, 'ko', ms = 10, mfc='w', mew=2)
    
    for i in range(len(values_eids_unordered)):
        vs = values_eids_unordered[i]
        fillcircle_eid = fillcircle_eids_unordered[i]
        acr_inds = np.nonzero(acronyms==acronyms_unordered[i])[0]
        assert len(acr_inds) == 1
        ind = acr_inds[0]
        # plt.plot( ind*np.ones(len(vs)) + 0.4 * (np.random.rand(len(vs))-0.5), 
        #          vs, 
        #         'ko', markersize=4 )
        for j in range(len(vs)):
            mfc = 'k' if fillcircle_eid[j] else 'none'
            plt.plot( ind + 0.4 * (np.random.rand()-0.5), 
                         vs[j], 
                         'ko', markersize=4 , mfc=mfc)
    plt.xticks(inds, labels=acronyms, rotation=90)
    xtls = fig.axes[0].get_xticklabels()
    [xtl.set_fontweight('extra bold') for xtl in xtls if xtl.get_text() in bolded_regions]
    #fig.axes[0].set_xticklabels(xtls)
    print('bar, x-axis tick labels', xtls)
    if not (ticks is None):
        plt.yticks(ticks[0], labels=ticks[1])
    #print(acronyms)
    plt.ylabel(ylab)
    if not (YMIN is None) and (YMAX is None):
        plt.ylim(YMIN, 
                 1.1*(np.max(np.concatenate(values_eids_unordered))-YMIN)+YMIN)
        
    if (YMIN is None) and not (YMAX is None):
        plt.ylim(YMAX - 1.1*(YMAX - np.min(np.concatenate(values_eids_unordered))),
                 YMAX)
    if not (YMIN is None) and not (YMAX is None):
        plt.ylim(YMIN, YMAX)
        
    plt.xlim(np.min(inds)-1, np.max(inds)+1)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return acronyms

preamb = 'decoding_results/summary/'

#%% stimside

file_all_results = preamb + '02-04-2023_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]

regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.region)
regions_FDRsig = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.null_median_of_medians)
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='stimside_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            sort_args=None, 
                            bolded_regions=regions_FDRsig)

#%% choice

file_all_results = preamb + '01-04-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]

regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.region)
regions_FDRsig = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.null_median_of_medians)
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='choice_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            YMAX=1.0,
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            #TOP_N=15,
                            sort_args=None, 
                            bolded_regions=regions_FDRsig)

#%% feedback

file_all_results = preamb + '01-04-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]

regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.region)
regions_FDRsig = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.null_median_of_medians)
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='feedback_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            YMAX=1.0,
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            sort_args=None, 
                            bolded_regions=regions_FDRsig)

#%% block

file_all_results = preamb + '04-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]

regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.region)
regions_FDRsig = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.null_median_of_medians)
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='block_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            YMAX=1.0,
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            sort_args=None, 
                            bolded_regions=regions_FDRsig)

#%% wheel-speed

file_all_results = preamb + '01-04-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]

regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.region)
regions_FDRsig = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.null_median_of_medians)
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='wheelspeed_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='$R^2$',
                            # ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            sort_args=None, 
                            bolded_regions=regions_FDRsig)

#%% wheel-velocity

file_all_results = preamb + '01-04-2023_decode_wheel-vel_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]

regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.region)
regions_FDRsig = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.null_median_of_medians)
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='wheelvel_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='$R^2$',
                            # ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            sort_args=None, 
                            bolded_regions=regions_FDRsig)