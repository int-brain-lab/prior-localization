import os
from fitplot_funs import plot_cellkerns, get_fullfit_cells
from export_data import trialinfo_to_df
import numpy as np
from oneibl import one
import matplotlib.pyplot as plt

# Get names of subjects for which fits are available
subjects = [x for x in os.listdir('./fits/') if os.path.isdir(f'./fits/{x}/')]
dates = {sub: os.listdir(f'./fits/{sub}/') for sub in subjects}
one = one.ONE()

for subject in subjects:
    if not os.path.exists(f'./fits/{subject}/plots/'):
        os.mkdir(f'./fits/{subject}/plots')
    for filename in dates[subject]:
        plotdir = f'./fits/{subject}/plots/{filename[:-6]}_run/'
        if not os.path.exists(plotdir):
            os.mkdir(plotdir)

        currfit = np.load(f'./fits/{subject}/{filename}', allow_pickle=True)

        spikes, clus = one.load(currfit['session_uuid'],
                                dataset_types=['spikes.times', 'spikes.clusters'])
        trialdf = trialinfo_to_df(currfit['session_uuid'])

        df = currfit['fits']
        # We need to get rid of the neutral bias zero contrast fits.
        # These have too few trials and are never fit.
        droprows = df.xs([0.5, 'Zero'], level=['bias', 'contr'], drop_level=False).index
        df.drop(index=droprows, inplace=True)
        fitcells = get_fullfit_cells(df)
        for cell in fitcells:
            fig, ax = plot_cellkerns(cell, [df, trialdf, spikes, clus, currfit['kern_length'],
                                     currfit['glm_binsize']])
            plt.savefig(plotdir + cell + '.png', DPI=1000)
            plt.close()
