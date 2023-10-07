#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:35:55 2023

@author: bensonb
"""
import os
import numpy as np
import pandas as pd
from ibllib.atlas.plots import reorder_data
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units

sns.set(font_scale=1.5)
sns.set_style('ticks')

MIN_BALACC = 0.45
MIN_R2 = -0.05

# get reference cluster dataframe
julias_clusters = bwm_units(ONE(base_url='https://openalyx.internationalbrainlab.org',
                                password='international'))
julias_clusters['sessreg'] = julias_clusters.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)
ref_clusters = julias_clusters[['uuids','sessreg']]

def filter_canonical(res_table):
    res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
    res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]
    return res_table

PATH_TO_ALLEN_COLOR_CSV = '../../allen_structure_tree.csv'
allen_color_data = np.genfromtxt(PATH_TO_ALLEN_COLOR_CSV, 
                                 delimiter=',',dtype=str)
yellows_to_convert = ['F0F080','FFFC91','ECE754','FFFDBC']

def hex2rgba(hex):
    if '' == hex:
        return (0,0,0,0)
    if hex in yellows_to_convert:
        hex = '767A3A'
    while len(hex) < 6:
        hex = '0'+hex
    assert len(hex) == 6
    return matplotlib.colors.to_rgba('#'+hex)

reg2rgba_dict = {allen_color_data[i,3]:
                 hex2rgba(allen_color_data[i,13]) for i in range(1,allen_color_data.shape[0])}


def bar_results(acronyms_unordered, 
                values_eids_unordered,
                nulls_eid_unordered, 
                fillcircle_eids_unordered = None,
                filename='test.png', 
                ylab='',
                ticks=None,
                FILE_PATH='./decoding_figures/SI/',
                YMIN=None,
                YMAX=None,
                bolded_regions=[],
                red_regions=[]):
    '''
    

    Parameters
    ----------
    acronyms_unordered : array
        region acronyms; all elements must be unique
    values_eids_unordered : array of arrays
        each element in array is an array of all sessions' values
        first dimension corresponds to regions in acronyms_unordered
    nulls_eids_unordered : array of arrays
        each element in array is an array of all sessions' median-null values
        first dimension corresponds to regions in acronyms_unordered
    fillcircle_eids_unordered : array of arrays
        each element in array is an array of all sessions' binary values to
        indicate whether the corresponding value in values_eid_unordered
        should be a filled black circle, 1, or an empty circle, 0.  The first 
        dimension corresponds to regions in acronyms_unordered
    filename : str, optional
        The default is 'test.png'.
    ylab : str, optional
        The default is ''.
    ticks : tuple, (list of ticks, list of labels), y-ticks to use
    FILE_PATH : str, optional
        The default is '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'.
    YMIN : float, optional
        The y-axis lower limit.  The default is None.
    YMIN : float, optional
        The y-axis lower limit.  The default is None.
    bolded_regions : array
        array of strings for each region which should be bolded
        in the x-axis.  default is empty array.
    red_regions : array
        array of strings for each region which should be made red
        in the x-axis.  default is empty array.
    
    Returns
    -------
    None

    '''
    # acronyms must be unique
    assert len(acronyms_unordered) == len(np.unique(acronyms_unordered))
    
    values_unordered = np.array([np.median(v) for v in values_eids_unordered])
    nulls_unordered = np.array([np.median(n) for n in nulls_eid_unordered])
    if fillcircle_eids_unordered is None:
        fillcircle_eids_unordered = np.array([np.ones(len(vs)) for vs in values_eids_unordered])
    
    # order acronyms, values, nulls
    acronyms, values = reorder_data(acronyms_unordered, values_unordered)
    acronyms_tmp, nulls = reorder_data(acronyms_unordered, nulls_unordered)
    assert np.all(acronyms == acronyms_tmp)
    assert len(values) == len(nulls)
        
    
    fig = plt.figure(figsize=(2.5 + int(len(acronyms)/4),5))
    plt.title('')
    
    # bars and white null dots
    inds = np.arange(len(acronyms))
    plt.bar(inds, values, 
            color=[reg2rgba_dict[r] for r in acronyms],
            )
    plt.plot(inds, nulls, 'ko', ms = 10, mfc='w', mew=2)
    
    # plot session-wise dots
    for i in range(len(values_eids_unordered)):
        vs = values_eids_unordered[i]
        fillcircle_eid = fillcircle_eids_unordered[i]
        acr_inds = np.nonzero(acronyms==acronyms_unordered[i])[0]
        assert len(acr_inds) == 1
        ind = acr_inds[0]
        for j in range(len(vs)):
            mfc = 'k'
            mec = 'k'
            mew = .5
            ms = 4 if fillcircle_eid[j] else 5
            mshape = 'o' if fillcircle_eid[j] else 'x'
            plt.plot( ind + 0.4 * (np.random.rand()-0.5), 
                         vs[j], 
                         mshape, markersize=ms , mfc=mfc, mec=mec, mew=mew)
            
    # x-axis ticks, labels, limits
    plt.xticks(inds, labels=acronyms, rotation=90)
    xtls = fig.axes[0].get_xticklabels()
    [xtl.set_fontweight('extra bold') for xtl in xtls if xtl.get_text() in bolded_regions]
    [xtl.set_color('red') for xtl in xtls if xtl.get_text() in red_regions]
    plt.xlim(np.min(inds)-1, np.max(inds)+1)
    
    # y-axis ticks, labels, limits
    if not (ticks is None):
        plt.yticks(ticks[0], labels=ticks[1])        
    plt.ylabel(ylab)
    if not (YMIN is None) and (YMAX is None):
        plt.ylim(YMIN, 
                 1.1*(np.max(np.concatenate(values_eids_unordered))-YMIN)+YMIN)
    if (YMIN is None) and not (YMAX is None):
        plt.ylim(YMAX - 1.1*(YMAX - np.min(np.concatenate(values_eids_unordered))),
                 YMAX)
    if not (YMIN is None) and not (YMAX is None):
        plt.ylim(YMIN, YMAX)
    
    # axes location, visualization, and saving
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8  # Adjust these values as desired
    ax = plt.gca()
    ax.set_position([left, bottom, width, height],
                    which = 'both')
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    SAVE_PATH = os.path.join(FILE_PATH, filename)
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    
    return

preamb = 'decoding_results/summary/'

#%% stimside

# load session-wise data and filter according to canonical set
file_all_results = preamb + '02-04-2023_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
res_table = filter_canonical(res_table)

# define decoding scores, grouped by regions.
#     values: list of lists
#            the i-th value is a list of per-session scores in region 
#            regions[i]
#     pvalues: list of lists
#            the i-th value is a list of per-session p-values for region 
#            regions[i]
#     values_sig: list of lists
#            the i-th value is a list of per-session binary-significant values 
#            at alpha=0.05 for region regions[i]
#     mednulls: list of list
#            the i-th value is a list of per-session null-distribution-median
#            in region regions[i]
regions = np.unique(res_table['region'])
get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])
assert MIN_R2 < np.min([np.min(v) for v in values])
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
pvalues = np.array([get_pvals(reg) for reg in regions])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])
get_mednull = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
mednulls = np.array([get_mednull(reg) for reg in regions])

# define regions which pass FDR signifiance at 0.01
comb_ps = [scipy.stats.combine_pvalues(ps, method='fisher')[1] for ps in pvalues]
fdrsig, comb_pscorr, _, _ = multipletests(comb_ps, 
                                 0.01, method='fdr_bh')
regions_FDRsig = regions[fdrsig]

acr_plotted = bar_results(regions, 
                            values,
                            mednulls, 
                            values_sig,
                            filename='stimside_bars.svg', 
                            YMIN=MIN_BALACC,
                            YMAX=1.0,
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            bolded_regions=regions_FDRsig)

#%% choice

# load session-wise data and filter according to canonical set
file_all_results = preamb + '01-04-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
res_table = filter_canonical(res_table)

# define decoding scores, grouped by regions.
#     values: list of lists
#            the i-th value is a list of per-session scores in region 
#            regions[i]
#     pvalues: list of lists
#            the i-th value is a list of per-session p-values for region 
#            regions[i]
#     values_sig: list of lists
#            the i-th value is a list of per-session binary-significant values 
#            at alpha=0.05 for region regions[i]
#     mednulls: list of list
#            the i-th value is a list of per-session null-distribution-median
#            in region regions[i]
regions = np.unique(res_table['region'])
get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])
assert MIN_R2 < np.min([np.min(v) for v in values])
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
pvalues = np.array([get_pvals(reg) for reg in regions])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])
get_mednull = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
mednulls = np.array([get_mednull(reg) for reg in regions])

# define regions which pass FDR signifiance at 0.01
comb_ps = [scipy.stats.combine_pvalues(ps, method='fisher')[1] for ps in pvalues]
fdrsig, comb_pscorr, _, _ = multipletests(comb_ps, 
                                 0.01, method='fdr_bh')
regions_FDRsig = regions[fdrsig]

acr_plotted = bar_results(regions, 
                            values,
                            mednulls, 
                            values_sig,
                            filename='choice_bars.svg', 
                            YMIN=MIN_BALACC,
                            YMAX=1.0,
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            bolded_regions=regions_FDRsig)

#%% feedback

# load session-wise data and filter according to canonical set
file_all_results = preamb + '01-04-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
res_table = filter_canonical(res_table)

# define decoding scores, grouped by regions.
#     values: list of lists
#            the i-th value is a list of per-session scores in region 
#            regions[i]
#     pvalues: list of lists
#            the i-th value is a list of per-session p-values for region 
#            regions[i]
#     values_sig: list of lists
#            the i-th value is a list of per-session binary-significant values 
#            at alpha=0.05 for region regions[i]
#     mednulls: list of list
#            the i-th value is a list of per-session null-distribution-median
#            in region regions[i]
regions = np.unique(res_table['region'])
get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])
assert MIN_R2 < np.min([np.min(v) for v in values])
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
pvalues = np.array([get_pvals(reg) for reg in regions])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])
get_mednull = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
mednulls = np.array([get_mednull(reg) for reg in regions])

# define regions which pass FDR signifiance at 0.01
comb_ps = [scipy.stats.combine_pvalues(ps, method='fisher')[1] for ps in pvalues]
fdrsig, comb_pscorr, _, _ = multipletests(comb_ps, 
                                 0.01, method='fdr_bh')
regions_FDRsig = regions[fdrsig]

acr_plotted = bar_results(regions, 
                            values,
                            mednulls, 
                            values_sig,
                            filename='feedback_bars.svg', 
                            YMIN=MIN_BALACC,
                            YMAX=1.0,
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            bolded_regions=regions_FDRsig)

#%% block

# load session-wise data and filter according to canonical set
file_all_results = preamb + '04-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
res_table = filter_canonical(res_table)

# define decoding scores, grouped by regions.
#     values: list of lists
#            the i-th value is a list of per-session scores in region 
#            regions[i]
#     pvalues: list of lists
#            the i-th value is a list of per-session p-values for region 
#            regions[i]
#     values_sig: list of lists
#            the i-th value is a list of per-session binary-significant values 
#            at alpha=0.05 for region regions[i]
#     mednulls: list of list
#            the i-th value is a list of per-session null-distribution-median
#            in region regions[i]
regions = np.unique(res_table['region'])
get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])
assert MIN_R2 < np.min([np.min(v) for v in values])
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
pvalues = np.array([get_pvals(reg) for reg in regions])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])
get_mednull = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
mednulls = np.array([get_mednull(reg) for reg in regions])

# define regions which pass FDR signifiance at 0.01
comb_ps = [scipy.stats.combine_pvalues(ps, method='fisher')[1] for ps in pvalues]
fdrsig, comb_pscorr, _, _ = multipletests(comb_ps, 
                                 0.01, method='fdr_bh')
regions_FDRsig = regions[fdrsig]

acr_plotted = bar_results(regions, 
                            values,
                            mednulls, 
                            values_sig,
                            filename='block_bars.svg', 
                            YMIN=MIN_BALACC,
                            YMAX=1.0,
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            bolded_regions=regions_FDRsig)

#%% wheel-speed

# load session-wise data and filter according to canonical set
file_all_results = preamb + '01-04-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
res_table = filter_canonical(res_table)

# define decoding scores, grouped by regions.
#     values: list of lists
#            the i-th value is a list of per-session scores in region 
#            regions[i]
#     pvalues: list of lists
#            the i-th value is a list of per-session p-values for region 
#            regions[i]
#     values_sig: list of lists
#            the i-th value is a list of per-session binary-significant values 
#            at alpha=0.05 for region regions[i]
#     mednulls: list of list
#            the i-th value is a list of per-session null-distribution-median
#            in region regions[i]
regions = np.unique(res_table['region'])
get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])
assert MIN_R2 < np.min([np.min(v) for v in values])
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
pvalues = np.array([get_pvals(reg) for reg in regions])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])
get_mednull = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
mednulls = np.array([get_mednull(reg) for reg in regions])

# define regions which pass FDR signifiance at 0.01
comb_ps = [scipy.stats.combine_pvalues(ps, method='fisher')[1] for ps in pvalues]
fdrsig, comb_pscorr, _, _ = multipletests(comb_ps, 
                                 0.01, method='fdr_bh')
regions_FDRsig = regions[fdrsig]

acr_plotted = bar_results(regions, 
                            values,
                            mednulls,
                            values_sig,
                            filename='wheelspeed_bars.svg', 
                            YMIN=MIN_R2,
                            YMAX=1.0,
                            ylab='$R^2$',
                            # ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            bolded_regions=regions_FDRsig)

#%% wheel-velocity

# load session-wise data and filter according to canonical set
file_all_results = preamb + '01-04-2023_decode_wheel-vel_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
res_table = filter_canonical(res_table)

# define decoding scores, grouped by regions.
#     values: list of lists
#            the i-th value is a list of per-session scores in region 
#            regions[i]
#     pvalues: list of lists
#            the i-th value is a list of per-session p-values for region 
#            regions[i]
#     values_sig: list of lists
#            the i-th value is a list of per-session binary-significant values 
#            at alpha=0.05 for region regions[i]
#     mednulls: list of list
#            the i-th value is a list of per-session null-distribution-median
#            in region regions[i]
regions = np.unique(res_table['region'])
get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])
assert MIN_R2 < np.min([np.min(v) for v in values])
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
pvalues = np.array([get_pvals(reg) for reg in regions])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])
get_mednull = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
mednulls = np.array([get_mednull(reg) for reg in regions])

# define regions which pass FDR signifiance at 0.01
comb_ps = [scipy.stats.combine_pvalues(ps, method='fisher')[1] for ps in pvalues]
fdrsig, comb_pscorr, _, _ = multipletests(comb_ps, 
                                 0.01, method='fdr_bh')
regions_FDRsig = regions[fdrsig]

acr_plotted = bar_results(regions, 
                            values,
                            mednulls,
                            values_sig,
                            filename='wheelvel_bars.svg', 
                            YMIN=MIN_R2,
                            YMAX=1.0,
                            ylab='$R^2$',
                            # ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            bolded_regions=regions_FDRsig)