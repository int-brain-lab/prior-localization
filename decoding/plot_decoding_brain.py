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
from ibllib.atlas import AllenAtlas
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
    
    acronyms, values = reorder_data(acronyms_unordered, values_unordered)
    acronyms_tmp, errs = reorder_data(acronyms_unordered, errs_unordered)
    assert np.all(acronyms == acronyms_tmp)
    assert len(values) == len(errs)
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
        plt.ylim(YMIN, 1.1*(np.max(values+errs)-YMIN)+YMIN)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()

# from ibllib.atlas import AllenAtlas
# ba=AllenAtlas()
# acronyms = ba.regions.acronym
# values = np.random.random((acronyms.shape[0]))
# filename = 'test_plotrandomr2s.png'
# plot_decoding_results(acronyms, values, filename)
