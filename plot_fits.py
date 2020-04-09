import os
from fitplot_funs import plot_cellkerns, get_fullfit_cells
from export_funs import trialinfo_to_df
import numpy as np
from oneibl import one
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import filterwarnings

# Get names of subjects for which fits are available
subjects = [x for x in os.listdir('./fits/') if os.path.isdir(f'./fits/{x}/')]
dates = {sub: os.listdir(f'./fits/{sub}/') for sub in subjects}
one = one.ONE()

for subject in subjects:
    if not os.path.exists(f'./fits/{subject}/plots/'):
        os.mkdir(f'./fits/{subject}/plots')
    for filename in dates[subject]:
        if filename.split('.')[-1] != 'p':
            continue
        if os.path.isdir(f'./fits/{subject}/{filename}'):
            continue
        print(f'Working on {subject} : {filename}')
        plotdir = f'./fits/{subject}/plots/{filename[:-6]}_run/'
        if not os.path.exists(plotdir):
            os.mkdir(plotdir)

        currfit = np.load(f'./fits/{subject}/{filename}', allow_pickle=True)
        probe_idx = int(filename[filename.index('probe') + 5])
        try:
            spikes, clus = one.load(currfit['session_uuid'],
                                    dataset_types=['spikes.times', 'spikes.clusters'])
        except ValueError:
            spikes = one.load(currfit['session_uuid'], dataset_types=['spikes.times'])[probe_idx]
            clus = one.load(currfit['session_uuid'], dataset_types=['spikes.clusters'])[probe_idx]
        trialdf = trialinfo_to_df(currfit['session_uuid'])

        df = currfit['fits']
        # We need to get rid of the neutral bias zero contrast fits.
        # These have too few trials and are never fit.
        droprows = df.xs([0.5, 'Zero'], level=['bias', 'contr'], drop_level=False).index
        df.drop(index=droprows, inplace=True)
        fitcells = get_fullfit_cells(df)
        if len(fitcells) == 0:
            print('No cells with all conds fit. Dropping zero contrast from reqs.')
            dropmorerows = df.xs('Zero', level='contr', drop_level=False).index
            df.drop(index=dropmorerows, inplace=True)
            fitcells = get_fullfit_cells(df)
        filterwarnings('ignore')
        for cell in tqdm(fitcells):
            if os.path.exists(plotdir + cell + '.png'):
                continue
            fig, ax = plot_cellkerns(cell, [df, trialdf, spikes, clus, currfit['kern_length'],
                                     currfit['glm_binsize']])
            plt.savefig(plotdir + cell + '.png', DPI=1000)
            plt.close()
