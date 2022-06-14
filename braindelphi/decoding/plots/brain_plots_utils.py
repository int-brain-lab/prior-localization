
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from ibllib.atlas import AllenAtlas
from ibllib.atlas.plots import plot_scalar_on_slice

from ibllib.atlas import AllenAtlas
ba = AllenAtlas(res_um=25) # res_um = 10 for better resolution

def format_results(results,correction='mean'):

    '''
    This function format the results of decoding into appropriate statistics.
    '''

    # mean over the run to remove the variability du to the cross-validation
    df_subject = results.groupby(['eid','probe','region','pseudo_id']).agg(np.mean)

    # isolate the true sessions regressions
    df_true = df_subject[ df_subject.index.get_level_values('pseudo_id') == -1]

    # and the pseudo sessions regressions
    df_pseudo = df_subject[ df_subject.index.get_level_values('pseudo_id') != -1]

    # compute the median of pseudo-sessions results over the pseudo_id for each regression
    if correction == 'median':
        print('median correction')
        df_pseudo_median = df_pseudo.groupby(['eid','probe','region']).agg(np.median)
    else :
        print('mean correction')
        df_pseudo_median = df_pseudo.groupby(['eid','probe','region']).mean()

    df_pseudo_median = df_pseudo_median.add_prefix('pseudo_')['pseudo_R2_test']
    
    # join pseudo regressions median results to each corresponding true regression results
    df_sess_reg = df_true.join(df_pseudo_median, how='outer')

    # for each regression, get the corrected results for R2 score
    df_sess_reg['corrected_R2_test'] = df_sess_reg["R2_test"] - df_sess_reg["pseudo_R2_test"]

    # average corrected results over all regressions for each region
    df_reg = df_sess_reg.groupby('region').mean()

    # compute the 95% quantile for R2 score - AT REGION LEVEL
    # first average the shift over all regressions for each region for each pseudosession
    df_pseudo_R2_reg = df_pseudo['R2_test'].groupby(['region','pseudo_id']).mean()
    #then compute 95% quantile for each region
    df_pseudo_R2_quantile = df_pseudo_R2_reg.groupby('region').quantile(0.95)
    df_pseudo_R2_quantile = df_pseudo_R2_quantile.rename('pseudo_95_quantile')
    df_reg = df_reg.join(df_pseudo_R2_quantile, how='outer')

    # compute the 95% quantile for R2 score - AT SESSION LEVEL
    df_pseudo_R2_sess_reg = df_pseudo['R2_test']
    #then compute 95% quantile for each region
    df_pseudo_R2_quantile_sess_reg = df_pseudo_R2_sess_reg.groupby(['region','eid','probe']).quantile(0.95)
    df_pseudo_R2_quantile_sess_reg = df_pseudo_R2_quantile_sess_reg.rename('pseudo_95_quantile')
    df_sess_reg = df_sess_reg.join(df_pseudo_R2_quantile_sess_reg, how='outer')
    
    return df_reg,df_sess_reg, df_pseudo

def shiftedColorMap(cmap, start=0, midpoint=0.0, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormaps dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=newcmap)

    return newcmap

def plot_region_variable_branwide(df,df_roi,var_name='R2_test',var_label='R2',title='',cmap_name='Blues',midpoint=0):
    
    # example w/ random value
    # acronyms = ba.regions.acronym
    # values = np.random.random((acronyms.shape[0]))

    acronyms_full = df.index.get_level_values(0).values
    
    acronyms = df_roi.index.get_level_values(0).values
    values = df_roi[var_name].to_numpy()

    mask = ~np.isnan(df_roi[var_name])

    acronyms = acronyms[mask]
    values = values[mask]

    fig, axes = plt.subplots(2,2,figsize=(5,5))

    cmap=plt.get_cmap(cmap_name)
    cmap = shiftedColorMap(cmap, start=0, midpoint=midpoint, stop=1.0, name='shiftedcmap')

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-1500, slice='top', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba,cmap=cmap, ax=axes[0,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-1500, slice='horizontal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=cmap,ax=axes[0,1])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=-1000, slice='sagittal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba,cmap=cmap, ax=axes[1,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        acronyms, values, coord=0, slice='coronal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba,cmap=cmap, ax=axes[1,1])
    ax.set_axis_off()

    fig.suptitle(title)

    # Plotting in grey the regions for which we didn't do any decoding.
    Beryl_acronyms = list(np.unique(ba.regions.acronym2acronym(ba.regions.acronym,mapping='Beryl')))
    no_reg_regions = list(set(Beryl_acronyms) - set(acronyms_full))
    blank_values = np.array( [1] * len(no_reg_regions) )
    no_reg_regions = np.array(no_reg_regions)

    my_cmap = mpl.colors.ListedColormap(['dimgray'], name = 'my_name')

    _, ax = plot_scalar_on_slice(
        no_reg_regions, blank_values, coord=-1500, slice='top', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba,cmap=my_cmap, ax=axes[0,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        no_reg_regions, blank_values, coord=-1500, slice='horizontal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba, cmap=my_cmap,ax=axes[0,1])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        no_reg_regions, blank_values, coord=-1000, slice='sagittal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba,cmap=my_cmap, ax=axes[1,0])
    ax.set_axis_off()

    _, ax = plot_scalar_on_slice(
        no_reg_regions, blank_values, coord=0, slice='coronal', mapping='Beryl', hemisphere='left', 
        background='boundary', brain_atlas=ba,cmap=my_cmap, ax=axes[1,1])
    ax.set_axis_off()
    
    fig.subplots_adjust(right=0.85)
    # lower left corner in [0.88, 0.3]
    # axes width 0.02 and height 0.4
    cb_ax = fig.add_axes([0.88, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(mappable=ax.images[0], cax=cb_ax)
    cb_ax.set_title(var_label)

    plt.draw()

    '''
    print(len(Beryl_acronyms),"regions in the beryl atlas")
    print(len(no_reg_regions),"regions with not enough data to perform decoding, in grey")
    print(len(acronyms_full),"regions with enough data to perform decoding")
    print(len(list(acronyms)),'regions from which we can significatively decode the prior')
    '''