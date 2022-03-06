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
from ibllib.atlas import AllenAtlas
from sklearn.metrics import r2_score
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


def brain_results(acronyms_unordered, values_unordered, 
                  filename, 
                FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/decoding_figures/',
                cmap='viridis',
                YMIN=None,
                value_title='$R^2$'):
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
    if not (YMIN is None):
        clevels = [YMIN, np.max(values)]
        if np.min(values) < YMIN:
            extend = 'min'
            values = np.maximum(values,YMIN)
        
    
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
    top_regions = acronyms[np.argsort(values)[::-1][:15]]
    top_regions_string = 'Top 15 Regions:\n   '+('\n   '.join(top_regions))
    fig.text(.38, .2, top_regions_string)
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return

def bar_results(acronyms_unordered, values_unordered, errs_unordered, 
                filename, 
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
    
def get_saved_data(results,result_index,
                              RESULTS_PATH,
                              SPECIFIC_DECODING,
                              return_number_of_active_neurons=False):
    subject = results.loc[result_index,'subject']
    eid = results.loc[result_index,'eid']
    probe = results.loc[result_index,'probe']
    region = results.loc[result_index,'region']
    data_path = os.path.join(RESULTS_PATH,
                             SPECIFIC_DECODING,
                             subject,
                             eid,
                             probe)
    names = [name for name in os.listdir(data_path) if '_'+region+'.' in name]
    
    assert len(names)==1
    data_name = names[0]
    data_df = pd.read_pickle(os.path.join(data_path,data_name))
    datafit_df = data_df['fit']
    preds = np.concatenate(datafit_df['predictions_test'])
    inds = np.concatenate(datafit_df['idxes_test'])
    preds = preds[np.argsort(inds)]
    inds = inds[np.argsort(inds)]
    assert len(np.unique(inds))==len(inds)
    assert np.max(inds)==len(inds)-1
    target = datafit_df['target']
    block_pLeft = datafit_df['pLeft_vec']
    block_pLeft = block_pLeft[datafit_df['mask']]
    
    if return_number_of_active_neurons:
        all_weights = np.concatenate(datafit_df['weights'])
        average_neurons_active = np.mean(all_weights>0.01) * data_df['N_units']
    
        return target, preds, block_pLeft, average_neurons_active
    return target, preds, block_pLeft

def aggregate_data(results, RESULTS_PATH, SPECIFIC_DECODING):
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
    all_accuracies = []
    all_r2s = []
    all_pvalues = []
    
    for result_index in results.index:
        target, preds, block_pLeft, actn = get_saved_data(results,
                                            result_index,
                                            RESULTS_PATH,
                                            SPECIFIC_DECODING,
                                            return_number_of_active_neurons=True)
        r2_value = r2_score(target,preds)
        null_r2s = np.array([results.loc[result_index,
                     'Rsquared_test_pseudo'+str(i)] for i in range(100)])
        p_value = np.mean(null_r2s > r2_value)
        
        assert results.loc[result_index,'Rsquared_test'] == r2_value
        
        all_pvalues.append(p_value)
        all_r2s.append(r2_value)
        all_accuracies.append(np.mean(1-np.abs(target-preds)))
        all_targets.append(target)
        all_preds.append(preds)
        all_block_pLeft.append(block_pLeft)
        all_actn.append(actn)
        all_regions.append(results.loc[result_index,'region'])
        all_eids.append(results.loc[result_index,'eid'])
        all_probes.append(results.loc[result_index,'probe'])
        
    all_pvalues = np.array(all_pvalues)
    all_r2s = np.array(all_r2s)
    all_accuracies = np.array(all_accuracies)
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_block_pLeft = np.array(all_block_pLeft)
    all_actn = np.array(all_actn)
    all_regions = np.array(all_regions)
    all_eids = np.array(all_eids)
    all_probes = np.array(all_probes)
    
    all_dict = {'p-value': all_pvalues,
                'r2': all_r2s,
                'accuracy': all_accuracies,
                'target': all_targets,
                'prediction': all_preds,
                'block_pLeft': all_block_pLeft,
                'active_neurons': all_actn,
                'region': all_regions,
                'eid': all_eids,
                'probe': all_probes}
    return all_dict