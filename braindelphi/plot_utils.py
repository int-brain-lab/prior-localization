#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:10:00 2022

@author: bensonb
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from ibllib.atlas import AllenAtlas, FlatMap
#from ibllib.atlas.plots import plot_scalar_on_flatmap
from ibllib.atlas.flatmaps import plot_swanson
from ibllib.atlas import BrainRegions
from ibllib.atlas.plots import reorder_data

def brain_SwansonFlat_results(acronyms, values, 
                  filename=None, 
                  cmap='viridis',
                  clevels=[None, None],
                  ticks=None,
                  extend=None,
                  value_title='',
                  FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/braindelphi/decoding_figures/'):
    '''

    Parameters
    ----------
    acronyms : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.
    filename : TYPE, optional
        DESCRIPTION. The default is None.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'viridis'.
    clevels : TYPE, optional
        DESCRIPTION. The default is [None, None].
    ticks : tuple, (tick_list, tick_label_list)
        DESCRIPTION. The default is None.
    extend : TYPE, optional
        DESCRIPTION. The default is None.
    value_title : TYPE, optional
        DESCRIPTION. The default is ''.
    FILE_PATH : TYPE, optional
        DESCRIPTION. The default is '/home/bensonb/IntBrainLab/prior-localization/braindelphi/decoding_figures/'.

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
    ax_swan = plot_swanson(acronyms, 
                    values, 
                    cmap=cmap,
                    empty_color = 'grey',
                    hemisphere='left',
                    br=BrainRegions())#,ax=axes[0:9])
    ax_swan.grid(False)
    # ax_swan.yticks([])
    # ax_swan.xticks([])
    
    fig.subplots_adjust(right=0.85)
    
    cb_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    cbar = plt.colorbar(mappable=ax_swan.images[0], cax=cb_ax, 
                        extend=extend)
    #cbar.set_ticks
    cb_ax.set_title(value_title)
    if not (ticks is None):
        cb_ax.set_yticklabels(ticks[1])
        cb_ax.set_yticks(ticks[0])
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

allen_color_data = np.genfromtxt('../../allen_structure_tree.csv', 
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

def bar_results(acronyms_unordered, values_eids_unordered, nulls_unordered, 
                filename='test.png', 
                ylab='',
                FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/braindelphi/decoding_figures/',
                YMIN=None,
                TOP_N=np.nan,
                POOL_PROTOCOL='median'):
    '''
    

    Parameters
    ----------
    acronyms_unordered : array
        region acronyms
    values_eids_unordered : array of arrays
        each element in array is an array of all sessions' values
        first dimension corresponds to regions in acronyms_unordered
    nulls_unordered : array
        the null value to plot as white circle on each region's bar.
        each element corresponds to regions in acronyms_unordered
    filename : str, optional
        DESCRIPTION. The default is 'test.png'.
    ylab : str, optional
        DESCRIPTION. The default is ''.
    FILE_PATH : str, optional
        DESCRIPTION. The default is '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'.
    YMIN : float, optional
        DESCRIPTION. The default is None.
    TOP_N : int, optional
        only plot the top n values. The default is np.nan.
    POOL_PROTOCOL : str, optional
        'median' or 'mean'.  how to do per-region pooling across sessions.
        The default is 'median'.
    Returns
    -------
    None.

    '''
    if POOL_PROTOCOL == 'median':
        values_unordered = np.array([np.median(vs) for vs in values_eids_unordered])
    elif POOL_PROTOCOL == 'mean':
        values_unordered = np.array([np.mean(vs) for vs in values_eids_unordered])
    else:
        raise ValueError('This value of POOL_PROTOCOL is not implemented.')
        
    if not np.isnan(TOP_N):
        v_to_sort_on = values_unordered*np.maximum(values_unordered-nulls_unordered,0)
        sinds = np.argsort(v_to_sort_on)[::-1][:TOP_N]
        acronyms_unordered = acronyms_unordered[sinds]
        if len(nulls_unordered.shape)==1:
            nulls_unordered = nulls_unordered[sinds]
        elif len(nulls_unordered.shape)==2:
            nulls_unordered = np.vstack((nulls_unordered[0,:][sinds],
                                         nulls_unordered[1,:][sinds],
                                         nulls_unordered[2,:][sinds]))
        values_eids_unordered = values_eids_unordered[sinds]
        values_unordered = values_unordered[sinds]
    
    acronyms, values = reorder_data(acronyms_unordered, values_unordered)
    if len(nulls_unordered.shape)==1:
        acronyms_tmp, nulls = reorder_data(acronyms_unordered, nulls_unordered)
        assert np.all(acronyms == acronyms_tmp)
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
    plt.figure(figsize=(8,2))
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
        acr_inds = np.nonzero(acronyms==acronyms_unordered[i])[0]
        assert len(acr_inds) == 1
        ind = acr_inds[0]
        plt.plot( ind*np.ones(len(vs)) + 0.4 * (np.random.rand(len(vs))-0.5), 
                     vs, 
                     'ko', markersize=4 )
    plt.xticks(inds, labels=acronyms, rotation=90)
    #print(acronyms)
    plt.ylabel(ylab)
    if not (YMIN is None):
        plt.ylim(YMIN, 
                 1.1*(np.max(np.concatenate(values_eids_unordered))-YMIN)+YMIN)
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return
