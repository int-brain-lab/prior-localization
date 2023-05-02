#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:10:00 2022

@author: bensonb
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
#from ibllib.atlas import AllenAtlas, FlatMap
#from ibllib.atlas.plots import plot_scalar_on_flatmap
from ibllib.atlas.flatmaps import plot_swanson
from ibllib.atlas import BrainRegions, AllenAtlas
from ibllib.atlas.plots import reorder_data
br = BrainRegions()

#%%
PATH_TO_ALLEN_COLOR_CSV = '../../allen_structure_tree.csv'

def acronym2name(acronym):
    return br.name[np.argwhere(br.acronym==acronym)[0]][0]

def get_full_region_name(acronyms):
    '''
    From Guido

    '''
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

def get_xy_vals(xy_table, eid, region):
    xy_vals = xy_table.loc[xy_table['eid_region']==f'{eid}_{region}']
    assert xy_vals.shape[0] == 1
    return xy_vals.iloc[0]

def get_predprob_vals(xy_table, eid, region):
    inds = list(xy_table[xy_table['eid_region']==f'{eid}_{region}'].index)
    assert len(inds) == 1
    ind = inds[0]
    # shape of (runs, neurons, trials)
    w_trials = np.empty((xy_table.loc[ind]['weights'].shape[0],
                          xy_table.loc[ind]['weights'].shape[3],
                          xy_table.loc[ind]['regressors'].shape[0]))
    w_trials[:] = np.nan
    for ri in range(10):
        for fi in range(5):
            w_trials[ri,:,xy_table.loc[ind]['idxes'][ri,fi]] = xy_table.loc[ind]['weights'][ri,fi,0,:]
    assert not np.any(np.isnan(w_trials))        
    
    itcp_trials = np.empty((xy_table.loc[ind]['intercepts'].shape[0], 
                          xy_table.loc[ind]['regressors'].shape[0]))
    itcp_trials[:] = np.nan
    for ri in range(10):
        for fi in range(5):
            itcp_trials[ri,xy_table.loc[ind]['idxes'][ri,fi]] = xy_table.loc[ind]['intercepts'][ri,fi,0]
    assert not np.any(np.isnan(itcp_trials))
    
    expvals = -np.einsum('rnt,tn->rt',
                          w_trials,
                          xy_table.loc[ind]['regressors'][:,0,:]) - itcp_trials
    predprobs = 1./(1+np.exp(expvals))
    # assert np.all(xy_table.loc[ind]['predictions'][:,:,0] == (predprobs>0.5))
    # assert np.all(xy_table.loc[ind]['predictions'][:,:,0] == predprobs)
    
    return predprobs.mean(axis=0)

def get_res_vals(res_table, eid, region):
    er_vals = res_table[(res_table['eid']==eid) & (res_table['region']==region)]
    assert len(er_vals)==1
    return er_vals.iloc[0]

def get_within_region_mean_var(res_table):
    regions = np.array([reg for reg in np.unique(res_table['region']) if not ((reg=='root') or (reg=='void'))])
    get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
    wi_means = [np.mean(get_vals(reg)) for reg in regions]
    wi_vars = [np.var(get_vals(reg)) for reg in regions]
    return wi_means, wi_vars

def comb_regs_df(res_table, 
                 USE_ALL_BERYL_REGIONS=True):
    '''
    combine all of the same regions in res_table and compute a set of 
    session-combined metrics per region.

    Parameters
    ----------
    res_table : pandas DataFrame,
        output of decoding pipeline which has scalar summaries of decoding
        results.  Different than xy_table which has multi-dimensional data.
    
    USE_ALL_BERYL_REGIONS : bool,
        if True, includes all beryl regions as rows of dataFrame.  Rows with 
        regions not in res_table have np.nan values.  Regions are ordered
        amongst rows using the ordering of regions in beryl.npy.
        if False, includes only those regions which are in res_table
        
    Returns
    -------
    comb_regs_data : pandas DataFrame

    '''
    
    regions = np.array([reg for reg in np.unique(res_table['region']) if not ((reg=='root') or (reg=='void'))])
    
    if USE_ALL_BERYL_REGIONS:
        all_regs = AllenAtlas().regions.id2acronym(np.load('../../beryl.npy'))
    else:
        all_regs = regions
    
    frac_sig_region = lambda reg: np.mean(np.array(res_table.loc[res_table['region']==reg,'p-value']<0.05))
    get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
    get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
    get_nulls = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
    get_valsminusnulls = lambda reg: get_vals(reg) - get_nulls(reg)
    get_nunits = lambda reg: np.array(res_table.loc[res_table['region']==reg,'n_units'])
    reg_comb_pval = lambda reg: scipy.stats.combine_pvalues(get_pvals(reg)
                                                            , method='fisher')[1]
    get_ms_reg = lambda reg: np.median(res_table.loc[(res_table['region']==reg) & (res_table['p-value']<0.05), 'score'])
    
    ps_uncorr = [reg_comb_pval(r) for r in regions]
    _, ps_corr, _, _ = multipletests(ps_uncorr, 0.05, method='fdr_bh')
    ps_uncorr = {regions[i]: ps_uncorr[i] for i in range(len(regions))}
    ps_corr = {regions[i]: ps_corr[i] for i in range(len(regions))}
    
    comb_regs_data = pd.DataFrame({'region': all_regs, 
              'combined_p-value': [ps_uncorr[r] if r in regions else np.nan for r in all_regs],
              'combined_sig': [ps_uncorr[r]<0.05 if r in regions else np.nan for r in all_regs],
              'combined_p-value_corr': [ps_corr[r] if r in regions else np.nan for r in all_regs],
              'combined_sig_corr': [ps_corr[r]<0.05 if r in regions else np.nan for r in all_regs],
              'n_sessions': [len(get_vals(r)) if r in regions else np.nan for r in all_regs],
              'n_units_mean': [np.mean(get_nunits(r)) if r in regions else np.nan for r in all_regs],
              'values_std': [np.std(get_vals(r)) if r in regions else np.nan for r in all_regs],
              'values_median': [np.median(get_vals(r)) if r in regions else np.nan for r in all_regs],
              'valuesminusnull_median': [np.median(get_valsminusnulls(r)) if r in regions else np.nan for r in all_regs],
              'frac_sig': [frac_sig_region(r) if r in regions else np.nan for r in all_regs],
              'values_median_sig': [get_ms_reg(r) if r in regions else np.nan for r in all_regs],
              'null_median_of_medians': [np.median(get_nulls(r)) if r in regions else np.nan for r in all_regs]})
    return comb_regs_data

def brain_SwansonFlat_results(acronyms, values, 
                  filename=None, 
                  cmap='viridis',
                  clevels=[None, None],
                  ticks=(None,None),
                  extend=None,
                  cbar_orientation='vertical',
                  value_title='',
                  FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/braindelphi/decoding_figures/'):
    '''

    Parameters
    ----------
    acronyms : array of strings
        
    values : array of values
        
    filename : string, optional
        The default is None.
    cmap : string, optional
        The default is 'viridis'.
    clevels : iterable of len 2, optional
        The default is [None, None].
    ticks : tuple, (tick_list, tick_label_list)
        colorbar ticks.  if not None, then cbar_orientation must be 'vertical',
        otherwise there is an issue
        The default is None.
    extend : string, optional
        can be 'max', 'min', 'both'
    cbar_orientation : string, optional
        'vertical' or 'horizontal', determines placement and orientation of 
        colorbar.  if ticks is not None, then use 'vertical', otherwise there 
        is an issue
        The default is 'vertical'
    value_title : string, optional
        The default is ''.
    FILE_PATH : string, optional
        The default is '/home/bensonb/IntBrainLab/prior-localization/braindelphi/decoding_figures/'.

    Returns
    -------
    None.

    '''
    
    print(clevels)
    if clevels[0] is None:
        print('min')
        clevels[0] = np.min(values)
    if clevels[1] is None:
        print('max')
        clevels[1] = np.max(values)
    values = np.clip(values, clevels[0], clevels[1])
    print(clevels, np.min(values), np.max(values))
    #extend = 'both'
    
    
    # plt.imshow([[0,1]], cmap=cmap, vmin=clevels[0], vmax=clevels[1], ax=axes[0,1])
    # plt.gca().set_visible(False)
    # plt.colorbar()# doesn't work
    # fig, axes = plt.subplots(1,10)
    fig = plt.figure()
    plt.title(value_title, fontsize=16)
    ax_swan = plot_swanson(acronyms, 
                    values, 
                    cmap=cmap, # empty_color = 'grey',
                    hemisphere='left',
                    br=BrainRegions())#,ax=axes[0:9])
    ax_swan.grid(False)
    # ax_swan.yticks([])
    # ax_swan.xticks([])
    
    fig.subplots_adjust(right=0.85)
    
    if cbar_orientation == 'vertical':
        cb_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    elif cbar_orientation == 'horizontal':
        cb_ax = fig.add_axes([0.2, 0.2, 0.6, 0.03])
    else:
        raise NotImplementedError('this color bar orientation is not implemented')
    cbar = plt.colorbar(mappable=ax_swan.images[0], cax=cb_ax, 
                        extend=extend, orientation=cbar_orientation)# ,
                        #ticks=None if ticks is None else ticks[0]
    cb_ax.tick_params(labelsize=14)
    #cbar.set_ticks
    #cb_ax.set_title(value_title)
    if not (ticks is None):
        cb_ax.set_yticks(ticks[0])
        cb_ax.set_yticklabels(ticks[1])
        
    #     print('ticks get', cbar)
    #     print('ticks 1', ticks[1])
        # cbar.ticks(ticks=ticks[0], labels=ticks[1])
        #cb_ax.set_yticklabels(ticks[1])
    #cb_ax.set_visible(False)
    # cbar.set_ticks([0,.2,.4,.6,.8,1])
    # cb_ax.set_title(value_title)
    #cbar.set_colorbar(extend='both')
    #plt.tight_layout()
    if not filename is None:
        SAVE_PATH = os.path.join(FILE_PATH, filename)
        plt.savefig(SAVE_PATH, dpi=1200)
    plt.show()
    return


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

def discretize_target(target_continuous,
                      edge = np.linspace(0,1,11)):
    
    target_discrete = np.copy(target_continuous)
    for i in range(len(edge)-1):
        if i == len(edge)-2: # last edge includes boundary
            target_discrete[(target_continuous >= edge[i])&
                               (target_continuous <= edge[i+1])] = .5*(edge[i]+edge[i+1])
        else:
            target_discrete[(target_continuous >= edge[i])&
                               (target_continuous < edge[i+1])] = .5*(edge[i]+edge[i+1])
    return target_discrete

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
                TOP_N=np.nan,
                POOL_PROTOCOL='median',
                sort_args=None,
                bolded_regions=None):
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
    fig = plt.figure(figsize=(8,2.5))
    plt.title(PLOT_TITLE)
    inds = np.arange(len(acronyms))
    plt.bar(inds, values, 
            color=[reg2rgba_dict[r] for r in acronyms],
            edgecolor='k',
            linewidth=1,
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
    if not (YMIN is None):
        plt.ylim(YMIN, 
                 1.1*(np.max(np.concatenate(values_eids_unordered))-YMIN)+YMIN)
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return acronyms

def sess2preds(ss_res, inverse_transf = None):
    '''
    

    Parameters
    ----------
    ss_res : TYPE
        DESCRIPTION.
    inverse_transf : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    preds : TYPE
        DESCRIPTION.
    targs : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.

    '''
    if inverse_transf is None:
        inverse_transf = lambda x: x
        
    all_test_predictions = []
    all_targets = []
    all_masks = []
    all_scores = []
    for i_run in range(len(ss_res['fit'])):
        # side, stim, act, _ = format_data_mut(result["fit"][i_run]["df"])
        mask = ss_res["fit"][i_run]["mask"]  # np.all(result["fit"][i_run]["target"] == stim[mask])
        # full_test_prediction = np.zeros(np.array(ss_res["fit"][i_run]["target"]).size)
        # for k in range(len(ss_res["fit"][i_run]["idxes_test"])):
        #     full_test_prediction[ss_res["fit"][i_run]['idxes_test'][k]] = ss_res["fit"][i_run]['predictions_test'][k]
        pt = ss_res["fit"][i_run]["predictions_test"]
        full_test_prediction = np.array([p[0] for p in pt])
        all_test_predictions.append(inverse_transf(full_test_prediction))
        all_targets.append(inverse_transf(ss_res["fit"][i_run]["target"]))
        all_masks.append(ss_res["fit"][i_run]["mask"])
        all_scores.append(ss_res["fit"][i_run]["scores_test_full"])
    
    preds = all_test_predictions[0]
    preds = np.mean(np.vstack(all_test_predictions),axis=0)
    fltg = lambda j: [all_targets[j][i][0] for i in range(len(all_targets[j]))]
    targs = np.array([np.array(fltg(j)) for j in range(len(all_targets))])
    assert np.all(np.abs(np.std(np.vstack(targs),axis=0))<1e-12)
    assert np.all(np.std(np.vstack(all_masks),axis=0)==0)
    targs = targs[0]
    mask = np.array(all_masks[0])+0
    return preds, targs, mask

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    * taken from matplotlib website * 
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks([])#, labels=row_labels)

    # Let the horizontal axes labeling appear on top/bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    * taken from matplotlib website * 
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def activity_and_decoding_weights(res_table, xy_table, 
                                  my_reg, save_path):
    
    xy_eids = [er.split('_')[0] for er in xy_table['eid_region'] if er.split('_')[1] == my_reg]
    # assert (len(xy_rgrs) == len(xy_trgs)) and (len(xy_rgrs)==len(xy_prds))
    # assert len(xy_rgrs) == len(xy_eids)
    Ns = [get_res_vals(res_table, xy_eids[i], my_reg)['n_units'] for i in range(len(xy_eids))]
    N = np.max(Ns)
    pvals = [get_res_vals(res_table, xy_eids[i], my_reg)['p-value'] for i in range(len(xy_eids))]
    sargs = np.argsort(pvals)
    xy_eids = np.array(xy_eids)
    xy_eids = xy_eids[sargs]
    
    
    Nsubplots = 2*len(xy_eids) + 1
    fig, axs = plt.subplots(Nsubplots, 
                            figsize=(int(0.7*(N+1)), 3.5*Nsubplots))
    
    axs[0].set_title(f'{acronym2name(my_reg)} ({my_reg})', 
                     fontsize=0.5*N)
    axs[0].hist(pvals)
    axs[0].set_ylabel('Count')
    axs[0].set_xlabel('P-values of all decoder in this region')
    axs[0].set_xlim(0,1)
    
    for ei in range(len(xy_eids)):
        xyi = ei*2 + 1
        my_eid = xy_eids[ei]
        xy_vals = get_xy_vals(xy_table, my_eid, my_reg)
        
        xy_rgr = np.squeeze(xy_vals['regressors']).T
        ws = np.squeeze(xy_vals['weights'])
        assert len(ws.shape) == 3
        xy_w = np.stack([np.ndarray.flatten(ws[:,:,i]) for i in range(ws.shape[2])]).T
        assert xy_w.shape[0] == 50
        xy_prd = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
        xy_trg = np.squeeze(xy_vals['targets'])
        xy_prm = xy_vals['params']
        #print(xy_rgr.shape)
        
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
                x.append([i, t, xy_rgr[i,t], xy_trg[t]])
        
        for t in range(xy_rgr.shape[1]):
            x.append([pr_xval, t, MAX_SPIKES*xy_prd[t], xy_trg[t]])
        
        df = pd.DataFrame(x, columns=['neuron',
                                      'trial',
                                      'spikes',
                                      'target'])
        mr = res_table.loc[(res_table['eid']==my_eid)&(res_table['region']==my_reg)]
        assert len(mr)==1
        mr = mr.iloc[0]
        nt = len(xy_trg)
        nt0 = len(xy_trg[xy_trg==0])
        nt1 = len(xy_trg[xy_trg==1])
        assert nt == (nt0 + nt1)
        axs[xyi].set_title(f'eid:{my_eid}, score:{mr["score"]:.3f}, p:{mr["p-value"]:.3f}, frac_w:{mr["frac_large_w"]:.3f}, gini_w:{mr["gini_w"]:.3f}, n_trials:{nt}, n_trials0:{nt0}, n_trials1:{nt1}',
                           fontsize=0.5*N)
        sns.violinplot(ax=axs[xyi], data=df, 
                       x="neuron", y="spikes", 
                       hue="target", split=True,
                       cut=0, linewidth=0)
        lxs = np.linspace(-0.4,0.4)
        lys = MAX_SPIKES*np.ones_like(lxs)
        axs[xyi].plot(lxs+pr_xval,lys,'r',lw=4)
        axs[xyi].plot(lxs+pr_xval,np.zeros_like(lxs),'r',lw=4)
        axs[xyi].text(pr_xval+0.5,MAX_SPIKES*(1),'Y*=1')
        axs[xyi].text(pr_xval+0.5,MAX_SPIKES*(0),'Y*=0')
        axs[xyi].text(pr_xval+0.6,MAX_SPIKES*(0.5), 
                      'Prediction of logistic decoder, Y*, \nbetween 0 and 1')
        tlabels = np.concatenate((np.arange(xy_n), np.array([pr_xval])))
        newlabels = ['Y*' if l==pr_xval else str(l) for l in tlabels]
        axs[xyi].set_xticks(tlabels)
        axs[xyi].set_xticklabels(newlabels)
        # align weights to violin plot
        # define heatmap edges relative to violin plot x-values 
        # as a fraction of total violin plot x-axis (c_0, c_f), 
        # then solve for the x-limits to choose in violin plot (Svec)
        # so that neuron numbers {0, 1... N-1} line up
        c_0, c_f = 0, .80
        M_inv = np.linalg.inv(np.array([[1-c_0, c_0],[1-c_f, c_f]]))
        Svec = np.matmul(M_inv, np.array([[-0.5],[xy_n+0.5]]))[:,0]
        axs[xyi].set_xlim(Svec[0],Svec[1])
        axs[xyi].legend(loc='upper left', title='Target')
        
        '''
        --------------------------------------------
        second plot: distribution of decoder weights
        --------------------------------------------
        '''
        xy_w_abs = np.abs(xy_w)
        
        prms = np.squeeze(xy_prm[:,:,:,1])
        assert len(prms.shape)==2
        assert prms.shape[0] == 10
        assert prms.shape[1] == 5
        prms = np.reshape(np.array(np.ndarray.flatten(prms), dtype=float), 
                          (50, 1))
        prms_plt = np.log10(prms)
        prms_plt = prms_plt - np.min(prms_plt)
        prms_plt = np.max(xy_w_abs)*prms_plt/np.max(prms_plt) if np.max(prms_plt)>0 else np.zeros_like(prms_plt)
        MW = np.hstack((xy_w_abs,prms_plt))
        
        w_ticks = np.arange(xy_n+1)
        im, cbar = heatmap(MW, np.arange(50), w_ticks, ax=axs[xyi+1],
                    cmap="YlGn", cbarlabel="Decoder weights (abs. value)",
                    aspect=xy_n/1500,
                    interpolation=None,
                    cbar_kw={'shrink': 0.7,
                             'location': 'right'})
        w_ticklabels = ['Decoder \nparams \n(log spacing)' if w == xy_n else str(w) for w in w_ticks]
        axs[xyi+1].set_xticks(w_ticks, labels=w_ticklabels)
        prm_mini = np.argmin(prms)
        prm_min = np.min(prms)
        prm_maxi = np.argmax(prms)
        prm_max = np.max(prms)
        axs[xyi+1].text(xy_n+0.5, prm_mini+1, 
                        f'--- min: {prm_min:.1e}',
                        fontsize=8)
        axs[xyi+1].text(xy_n+0.5, prm_maxi+1, 
                        f'--- max: {prm_max:.1e}',
                        fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path,dpi=100)
        
        return axs
    
