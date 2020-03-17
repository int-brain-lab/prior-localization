# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import brainbox.plot as bbp
import seaborn as sns


def err_wght_sync(errors, weights):
    if len(weights) % len(errors) != 0:
        raise ValueError(f'Weight len {len(weights)} isn\'t a full multiple'
                         f'of stats len {len(errors)}. '
                         'Make sure when fitting that the binsize and number of weights results'
                         ' in a whole-number multiple of kernel length and parameters.')
    mult = np.round(len(weights) / len(errors)).astype(int)
    newerr = np.array([[x] * mult for x in errors]).flatten()
    return newerr


def get_fullfit_cells(df):
    cells = sorted(list(set(df.index.get_level_values('cell_name'))))
    numfits = len(df.loc[cells[0]])
    keepcells = []
    for cell in cells:
        if sum(np.isfinite(df.loc[cell].biasStat)) == numfits:
            keepcells.append(cell)
    return keepcells


def plot_cellkerns(cell, data):
    df, tdf, spikes, clus, kern_length, glm_binsize = data
    fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
    contrasts = set(df.index.get_level_values('contr'))
    biases = list(set(df.index.get_level_values('bias')))
    cols = sns.color_palette('Paired')
    tstamps = np.arange(0, kern_length, glm_binsize)
    kernmap = {'stimOn': 'stimOn_times', 'fdbck': 'feedback_times'}
    for i, side in enumerate(['Left', 'Right']):
        for j, kern in enumerate(['stimOn', 'fdbck']):
            currc = 0
            for bias in biases:
                for contr in contrasts:
                    if (bias == 0.5) & (contr == 'Zero'):
                        continue
                    if contr == 'Zero':
                        filt1 = tdf[f'contrast{side}'] == 0
                    else:
                        filt1 = np.isfinite(tdf[f'contrast{side}'])
                    filt2 = tdf['probabilityLeft'] == bias
                    event_t = tdf[filt1 & filt2][kernmap[kern]]
                    bbp.peri_event_time_histogram(spikes, clus, event_t, int(cell[4:]), t_before=0,
                                                  t_after=0.6, ax=axes[2 * j, i], error_bars='sem',
                                                  pethline_kwargs={'color': cols[currc], 'lw': 2},
                                                  errbar_kwargs={'color': cols[currc],
                                                                 'alpha': 0.2})
                    currfit = df.loc[cell, side, bias, contr][kern]
                    currerr = df.loc[cell, side, bias, contr][kern + 'Stat']
                    assert tstamps.shape[0] == len(currfit), f'{cell} {side} {bias} {contr}'
                    newerr = err_wght_sync(currerr, currfit)
                    axes[2 * j + 1, i].errorbar(tstamps, currfit, newerr, color=cols[currc],
                                                label=f'bias {bias} contrast {contr}')
                    currc += 1
            axes[2 * j, i].set_title(f'{side} stimulus {kern} PSTH')
            newlims = axes[2 * j, i].get_ylim()
            axes[2 * j, i].set_ylim([newlims[0], newlims[1] * 1.3])  # Make sure get full PSTH
            axes[2 * j + 1, i].set_title(f'{side} stimulus, {kern} kernel')
            axes[2 * j + 1, i].legend()
    plt.tight_layout()
    return fig, axes
