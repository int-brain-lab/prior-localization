#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:41:26 2022

@author: bensonb
"""

#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from ibllib.atlas import AllenAtlas
ba=AllenAtlas()

import sys
PLOTUTILS_PATH = '/home/bensonb/IntBrainLab/paper-brain-wide-map/decoding/'
if not PLOTUTILS_PATH in sys.path:
    sys.path.insert(0, PLOTUTILS_PATH)
from utils import plot_scalar_on_slice

def plot_decoding_results(acronyms, values, filename, 
                FILE_PATH='/home/bensonb/IntBrainLab/prior-localization/decoding_figures/',
                cmap='viridis'):
    PLOT_TITLE = ''
    SAVE_PATH = FILE_PATH + filename
    # cmap = 'viridis'#'purples','blues','greens','oranges','reds'
    
    extend=None
    if np.min(values) < 0:
        extend = 'min'
    values = np.maximum(values,0)
    
    fig, axes = plt.subplots(2,2)
        
    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-1000, slice='top', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, ax=axes[0,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-3000, slice='horizontal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, ax=axes[1,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-1000, slice='sagittal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, ax=axes[0,1])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-500, slice='coronal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap, ax=axes[1,1])
    ax.set_axis_off()

    fig.suptitle(PLOT_TITLE)

    fig.subplots_adjust(right=0.85)
    # lower left corner in [0.88, 0.3]
    # axes width 0.02 and height 0.4
    cb_ax = fig.add_axes([0.88, 0.3, 0.02, 0.4])
    
    cbar = plt.colorbar(mappable=ax.images[0], cax=cb_ax, extend=extend)
    # cbar.set_ticks([0,.2,.4,.6,.8,1])
    cb_ax.set_title('$R^2$')
    top_regions = acronyms[np.argsort(values)[::-1][:15]]
    top_regions_string = 'Top 15 Regions:\n   '+('\n   '.join(top_regions))
    fig.text(.38, .2, top_regions_string)
    plt.savefig(SAVE_PATH, dpi=600)
    plt.show()
    return

# from ibllib.atlas import AllenAtlas
# ba=AllenAtlas()
# acronyms = ba.regions.acronym
# values = np.random.random((acronyms.shape[0]))
# filename = 'test_plotrandomr2s.png'
# plot_decoding_results(acronyms, values, filename)
