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
        if sum(np.isfinite(df.loc[cell].priorStat)) == numfits:
            keepcells.append(cell)
    return keepcells


def plot_cellkerns(cell, data):
    df, tdf, spikes, clus, kern_length, glm_binsize = data
    fig, axes = plt.subplots(5, 2, figsize=(15, 20), sharex=True)
    contrasts = set(df.index.get_level_values('contr'))
    biases = list(set(df.index.get_level_values('bias')))
    cols = sns.color_palette('Paired')
    tstamps = np.arange(0, kern_length, glm_binsize)
    kernmap = {'stimOn': 'stimOn_times', 'fdbck': 'feedback_times'}
    for i, side in enumerate(['Left', 'Right']):
        priorfactors = []
        priorstds = []
        priorlabels = []
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

                for j, kern in enumerate(['stimOn', 'fdbck']):
                    event_t = tdf[filt1 & filt2][kernmap[kern]]
                    oldmax = axes[2 * j, i].get_ylim()[1]
                    bbp.peri_event_time_histogram(spikes, clus, event_t, int(cell[4:]), t_before=0,
                                                  t_after=0.6, ax=axes[2 * j, i], error_bars='sem',
                                                  pethline_kwargs={'color': cols[currc], 'lw': 2},
                                                  errbar_kwargs={'color': cols[currc],
                                                                 'alpha': 0.2})
                    if oldmax > axes[2 * j, i].get_ylim()[1]:
                        axes[2 * j, i].set_ylim([0, oldmax])
                    currfit = df.loc[cell, side, bias, contr][kern]
                    currerr = df.loc[cell, side, bias, contr][kern + 'Stat']
                    assert tstamps.shape[0] == len(currfit), f'{cell} {side} {bias} {contr}'
                    newerr = err_wght_sync(currerr, currfit)
                    axes[2 * j + 1, i].errorbar(tstamps, currfit, newerr, color=cols[currc],
                                                label=f'bias {bias} contrast {contr}')
                    currc += 1
                    axes[2 * j, i].set_title(f'{side} stimulus {kern} PSTH')
                    axes[2 * j + 1, i].set_title(f'{side} stimulus, {kern} kernel')
                    axes[2 * j + 1, i].legend()
                priorfactors.append(df.loc[cell, side, bias, contr]['prior'][0])
                priorstds.append(df.loc[cell, side, bias, contr]['priorStat'])
                priorlabels.append(cell + side + str(bias) + ' mod')
        xmin, xmax = axes[0, i].get_xlim()
        edgestep = (xmax - xmin) / 5
        edges = np.arange(xmin, xmax - edgestep + 1e-3, edgestep)
        axes[4, i].bar(edges, priorfactors, yerr=np.array(priorstds) * 2, tick_label=priorlabels,
                       width=edgestep * 0.9, align='edge')
    plt.tight_layout()
    return fig, axes
