# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def err_wght_sync(errors, weights):
    if len(weights) % len(errors) != 0:
        raise ValueError('Weight len isn\'t a full multiple of stats len. '
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
    df, kern_length, glm_binsize = data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    contrasts = set(df.index.get_level_values('contr'))
    biases = list(set(df.index.get_level_values('bias')))
    tstamps = np.arange(0, kern_length, glm_binsize)
    for i, side in enumerate(['Left', 'Right']):
        for j, kern in enumerate(['stimOn', 'fdbck']):
            for contr in contrasts:
                for bias in biases:
                    if (bias == 0.5) & (contr == 'Zero'):
                        continue
                    currfit = df.loc[cell, side, bias, contr][kern]
                    currerr = df.loc[cell, side, bias, contr][kern + 'Stat']
                    assert tstamps.shape[0] == len(currfit), f'{cell} {side} {bias} {contr}'
                    newerr = err_wght_sync(currerr, currfit)
                    axes[j, i].errorbar(tstamps, currfit, np.sqrt(newerr),
                                        label=f'bias {bias} contrast {contr}')
            axes[j, i].set_title(f'{side} stimulus, {kern} kernel')
            axes[j, i].legend()
    return fig, axes
