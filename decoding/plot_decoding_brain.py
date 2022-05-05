#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:41:26 2022

@author: bensonb
"""

#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
from ibllib.atlas import AllenAtlas, FlatMap
from ibllib.atlas.plots import plot_scalar_on_flatmap
from sklearn.metrics import r2_score

# res = 25
# flmap = FlatMap(flatmap='dorsal_cortex', res_um=res)
# Plot flatmap at depth = 0
# flmap.plot_flatmap(int(0 / res))

ba=AllenAtlas()

import sys
import os
PLOTUTILS_PATH = '/home/bensonb/IntBrainLab/paper-brain-wide-map/decoding/'
if not PLOTUTILS_PATH in sys.path:
    sys.path.insert(0, PLOTUTILS_PATH)
PLOTUTILS_PATH = '/home/users/bensonb/international-brain-lab/paper-brain-wide-map/decoding/' # for cluster
if not PLOTUTILS_PATH in sys.path:
    sys.path.insert(0, PLOTUTILS_PATH)
from utils import plot_scalar_on_slice
from ibllib.atlas.plots import reorder_data

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

def brain_cortex_results(acronyms, values, 
                  filename=None, 
                  cmap='viridis',
                  clevels=[None, None],
                  FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'):

    # Plot region values on the left hemisphere at depth=0um overlaid on boundary image using Allen mapping
    # fig, ax = plot_scalar_on_flatmap(acronyms, values, depth=0, mapping='Allen', hemisphere='left', background='boundary',
    #                             cmap='viridis', flmap_atlas=FlatMap(flatmap='dorsal_cortex', res_um=25))
    # Plot two column region values on the both hemispheres at depth=0um on boundary image using Allen mapping
    if clevels[0] is None:
        clevels[0] = np.min(values)
    if clevels[1] is None:
        clevels[1] = np.max(values)
    
    fig, ax = plot_scalar_on_flatmap(acronyms, values, depth=0, 
                                     mapping='Beryl', hemisphere='left',
                                     background='boundary', 
                                     cmap=cmap, 
                                     clevels=clevels,
                                     show_cbar = False,
                                     flmap_atlas=FlatMap(flatmap='dorsal_cortex', res_um=25))
    
    #plt.colorbar() doesn't work
    #cbar.set_colorbar(extend='both')
    plt.grid(False)
    plt.yticks([])
    plt.xticks([])
    if not filename is None:
        SAVE_PATH = os.path.join(FILE_PATH, filename)
        plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return
    
def brain_results(acronyms_unordered, values_unordered, 
                  filename, 
                FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/decoding_figures/',
                cmap='viridis',
                YMIN=None,
                YMAX=None,
                value_title='$R^2$',
                TOP_N_TEXT=np.nan):
    '''
    

    Parameters
    ----------
    acronyms_unordered : TYPE
        DESCRIPTION.
    values_unordered : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.
    FILE_PATH : TYPE, optional
        DESCRIPTION. The default is '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'viridis'.
    YMIN : TYPE, optional
        DESCRIPTION. The default is None.
    YMAX : TYPE, optional
        DESCRIPTION. The default is None.
    value_title : TYPE, optional
        DESCRIPTION. The default is '$R^2$'.

    Returns
    -------
    None.

    '''
    acronyms, values = reorder_data(acronyms_unordered, values_unordered)
    
    PLOT_TITLE = ''
    SAVE_PATH = os.path.join(FILE_PATH, filename)
    # cmap = 'viridis'#'purples','blues','greens','oranges','reds'
    
    extend = None
    clevels = None
    if (not (YMIN is None)) and (not (YMAX is None)):
        clevels = [YMIN, YMAX]
        if (np.min(values) < YMIN) and (np.max(values) > YMAX):
            extend = 'both'
            values = np.minimum(np.maximum(values,YMIN),YMAX)
        elif np.min(values) < YMIN:
            extend = 'min'
            values = np.maximum(values,YMIN)
        elif np.max(values) > YMAX:
            extend = 'max'
            values = np.minimum(values,YMAX)
    elif not (YMIN is None):
        clevels = [YMIN, np.max(values)]
        if np.min(values) < YMIN:
            extend = 'min'
            values = np.maximum(values,YMIN)
    elif not (YMAX is None):
        clevels = [np.min(values), YMAX]
        if np.max(values) > YMAX:
            extend = 'max'
            values = np.minimum(values,YMAX)

    
    fig, axes = plt.subplots(2,2)
        
    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-1000, slice='top', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, clevels=clevels, ax=axes[0,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-3000, slice='horizontal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, clevels=clevels, ax=axes[1,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-1000, slice='sagittal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, clevels=clevels, ax=axes[0,1])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-500, slice='coronal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, clevels=clevels, ax=axes[1,1])
    ax.set_axis_off()

    fig.suptitle(PLOT_TITLE)

    fig.subplots_adjust(right=0.85)
    # lower left corner in [0.88, 0.3]
    # axes width 0.02 and height 0.4
    cb_ax = fig.add_axes([0.88, 0.3, 0.02, 0.4])
    
    cbar = plt.colorbar(mappable=ax.images[0], cax=cb_ax, extend=extend)
    # cbar.set_ticks([0,.2,.4,.6,.8,1])
    cb_ax.set_title(value_title)
    if not np.isnan(TOP_N_TEXT):
        TOP_N_TEXT = int(TOP_N_TEXT)
        top_regions = acronyms[np.argsort(values)[::-1][:TOP_N_TEXT]]
        top_regions_string = 'Top %d Regions:\n   '%TOP_N_TEXT \
                                        +('\n   '.join(top_regions))
        fig.text(.38, .2, top_regions_string)
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return clevels

def bar_results_basic(acronyms_unordered, values_unordered, errs_unordered=None, 
                filename='test.png', 
                ylab='',
                FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/decoding_figures/',
                YMIN=None):
    '''
    

    Parameters
    ----------
    acronyms_unordered : array
        DESCRIPTION.
    values_unordered : array
        DESCRIPTION.
    errs_unordered : array of 1 or 2 dimensions
        1 dimensional: size of error bars associated with values (equal size above and below value), 
        2 dimensional: dimension 1 is size 2, (errors below value, errors above value)
        if None, then all errors are zero
    filename : TYPE
        DESCRIPTION.
    ylab : str, optional
        DESCRIPTION. The default is ''.
    FILE_PATH : str, optional
        DESCRIPTION. The default is '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'.
    YMIN : float or int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    if errs_unordered is None:
        errs_unordered = np.zeros(len(values_unordered))
    assert len(errs_unordered.shape)==1 or len(errs_unordered.shape)==2
    acronyms, values = reorder_data(acronyms_unordered, values_unordered)
    if len(errs_unordered.shape) == 1:
        acronyms_tmp, errs = reorder_data(acronyms_unordered, errs_unordered)
        assert np.all(acronyms == acronyms_tmp)
        assert len(values) == len(errs)
        errs_high = errs
    else:
        acronyms_tmpp, errs_m = reorder_data(acronyms_unordered, errs_unordered[0,:])
        acronyms_tmpm, errs_p = reorder_data(acronyms_unordered, errs_unordered[1,:])
        assert np.all(acronyms == acronyms_tmpm)
        assert np.all(acronyms == acronyms_tmpp)
        assert len(values) == len(errs_m)
        assert len(values) == len(errs_p)
        errs = np.vstack((errs_m, errs_p))
        errs_high = errs_p
        
    PLOT_TITLE = ''
    SAVE_PATH = os.path.join(FILE_PATH, filename)
    plt.figure(figsize=(10,3))
    plt.title(PLOT_TITLE)
    inds = np.arange(len(acronyms))
    plt.bar(inds, values, 
            color=[reg2rgba_dict[r] for r in acronyms],
            yerr=errs)
    plt.xticks(inds, labels=acronyms, rotation=90)
    plt.ylabel(ylab)
    if not (YMIN is None):
        plt.ylim(YMIN, 1.1*(np.max(values+errs_high)-YMIN)+YMIN)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return

def bar_results(acronyms_unordered, values_eids_unordered, nulls_unordered, 
                filename='test.png', 
                ylab='',
                FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/decoding_figures/',
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
        corresponding to the regions in acronyms_unordered
    nulls_unordered : array
        the null value to plot as white circle on each region's bar
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
        sinds = np.argsort(values_unordered)[::-1][:TOP_N]
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
    
def get_saved_data(results,result_index,
                              RESULTS_PATH,
                              SPECIFIC_DECODING,
                              return_number_of_active_neurons=False,
                              get_probabilities=False):
    '''
    

    Parameters
    ----------
    results : TYPE
        DESCRIPTION.
    result_index : TYPE
        DESCRIPTION.
    RESULTS_PATH : TYPE
        DESCRIPTION.
    SPECIFIC_DECODING : TYPE
        DESCRIPTION.
    return_number_of_active_neurons : TYPE, optional
        DESCRIPTION. The default is False.
    get_probabilities : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    preds
        predictions of regression. if get_probabilities, then tuple 
        (predictions, probabilities), where probabilities are the probability 
        outputs of the classifier for the available classes

    '''
    subject = results.loc[result_index,'subject']
    eid = results.loc[result_index,'eid']
    probe = results.loc[result_index,'probe']
    region = results.loc[result_index,'region']
    data_path = os.path.join(RESULTS_PATH,
                             SPECIFIC_DECODING,
                             subject,
                             eid,
                             probe)
    masks = results.loc[result_index,'mask']
    names = [name for name in os.listdir(data_path) if '_'+region+'.' in name]
    
    assert len(names)==1
    data_name = names[0]
    data_df = pd.read_pickle(os.path.join(data_path,data_name))
    datafit_df = data_df['fit']
    inds = np.concatenate(datafit_df['idxes_test'])
    preds = np.concatenate(datafit_df['predictions_test'])
    preds = preds[np.argsort(inds)]
    if get_probabilities:
        probs = np.concatenate(datafit_df['probabilities_test'])
        probs = probs[np.argsort(inds)]
        preds = (np.copy(preds), np.copy(probs))
    inds = inds[np.argsort(inds)]
    assert len(np.unique(inds))==len(inds)
    assert np.max(inds)==len(inds)-1
    target = datafit_df['target']
    block_pLeft = datafit_df['pLeft_vec']
    block_pLeft = block_pLeft[datafit_df['mask']]
    
    if return_number_of_active_neurons:
        all_weights = np.concatenate(datafit_df['weights'])
        average_neurons_active = data_df['N_units']#np.mean(all_weights>0.01) * 
    
        return target, preds, block_pLeft, masks, average_neurons_active
    return target, preds, block_pLeft, masks

def aggregate_data(results, RESULTS_PATH, SPECIFIC_DECODING, 
                   get_probabilities=False):
    '''
    assumes 100 pseudo runs

    Parameters
    ----------
    results : pandas dataframe
        DESCRIPTION.
    RESULTS_PATH : str
        DESCRIPTION.
    SPECIFIC_DECODING : str
        DESCRIPTION.

    Returns
    -------
    all_dict : dictionary
        DESCRIPTION.

    '''
    all_eids = []
    all_probes = []
    all_regions = []
    all_targets = []
    all_preds = []
    all_block_pLeft = []
    all_actn = []
    all_scores = []
    all_null_scores = []
    all_pvalues = []
    all_masks = []
    if get_probabilities:
        all_probs = []
    
    for result_index in results.index:
        target, preds, block_pLeft, masks, actn = get_saved_data(results,
                                            result_index,
                                            RESULTS_PATH,
                                            SPECIFIC_DECODING,
                                            return_number_of_active_neurons=True,
                                            get_probabilities=get_probabilities)
        if get_probabilities:
            preds, probs = preds
            all_probs.append(probs[:,1])
            
        score = results.loc[result_index,'Score_test']
        
        null_scores = np.array([results.loc[result_index,
                     'Score_test_pseudo'+str(i)] for i in range(100)])
        p_value = np.mean(null_scores > score)
        
        # score = r2_score(target,preds)
        # assert results.loc[result_index,'Score_test'] == r2_value
        
        all_pvalues.append(p_value)
        all_scores.append(score)
        all_null_scores.append(null_scores)
        all_targets.append(target)
        all_preds.append(preds)
        all_block_pLeft.append(block_pLeft)
        all_actn.append(actn)
        all_regions.append(results.loc[result_index,'region'])
        all_eids.append(results.loc[result_index,'eid'])
        all_probes.append(results.loc[result_index,'probe'])
        all_masks.append(masks)
    
    all_dict = {'p-value': np.array(all_pvalues),
                'score': np.array(all_scores),
                'null_scores': np.array(all_null_scores),
                'target': np.array(all_targets),
                'prediction': np.array(all_preds),
                'block_pLeft': np.array(all_block_pLeft),
                'active_neurons': np.array(all_actn),
                'region': np.array(all_regions),
                'eid': np.array(all_eids),
                'probe': np.array(all_probes),
                'mask': np.array(all_masks)}
    if get_probabilities:
        all_dict['probability'] = np.array(all_probs)
        
    return all_dict