"""
Loop over eids and concatenate their information to form a block of data that we can sample from
to create null distributions of certain variables.
"""

import argparse
from brainbox.io.one import SessionLoader
import numpy as np
from one.api import ONE
import pandas as pd
from pathlib import Path

from brainwidemap import bwm_query, load_trials_and_mask
from prior_localization.functions.behavior_targets import add_behavior_to_df


def run_main(args):

    # collect args
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    target = args.target

    filename = save_dir.joinpath(f'imposterSessions_{target}.pqt')

    # TODO: what is the best way to pass these params?
    min_rt = 0.08
    max_rt = 2.0
    if target in ['wheel-speed', 'wheel-velocity']:
        align_event = 'firstMovement_times'
        time_window = (-0.2, 1.0)
        binsize = 0.02

    # ephys sessions from one of 12 templates
    one = ONE(base_url='https://openalyx.internationalbrainlab.org')  # , mode='local')
    bwm_df = bwm_query(freeze='2022_10_bwm_release')
    eids = bwm_df['eid'].unique()

    # basic columns that we want to keep
    columns = [
        'probabilityLeft',
        'contrastRight',
        'feedbackType',
        'choice',
        'contrastLeft',
        'eid',
        'template_sess',
        'firstMovement_times',
        'goCue_times',
        'stimOn_times',
        'feedback_times',
    ]

    # add additional columns if necessary
    add_behavior_col = target not in ['pLeft', 'signcont', 'feedback', 'choice']
    if add_behavior_col:
        columns += [target]

    all_trials_df = []
    for i, eid in enumerate(eids):

        print('%i: %s' % (i, eid))
        try:
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_trials()
            trials_df = sess_loader.trials
        except Exception as e:
            print('ERROR LOADING TRIALS DF')
            print(e)
            continue

        if add_behavior_col:

            # find bad trials
            _, trials_mask = load_trials_and_mask(
                one=one, eid=eid, sess_loader=sess_loader, min_rt=min_rt, max_rt=max_rt,
                min_trial_len=None, max_trial_len=None,
                exclude_nochoice=True, exclude_unbiased=False,
            )

            # add target data to dataframe
            intervals = np.vstack([
                sess_loader.trials[align_event] + time_window[0],
                sess_loader.trials[align_event] + time_window[1]
            ]).T
            trials_df, trials_mask = add_behavior_to_df(
                sess_loader, target, intervals, binsize, interval_len=time_window[1] - time_window[0],
                mask=trials_mask,
            )

        if trials_df is None:
            continue
        else:
            # add metadata
            trials_df.loc[:, 'eid'] = eid
            trials_df.loc[:, 'trial_id'] = trials_df.index
            trials_df.loc[:, 'template_sess'] = i
            # mask out "bad" trials
            trials_df = trials_df[trials_mask]
            # update
            all_trials_df.append(trials_df)

    all_trials_df = pd.concat(all_trials_df)

    # save imposter sessions
    all_trials_df[columns].to_parquet(filename)


if __name__ == '__main__':
    """python create_imposter_df.py --target=wheel-speed --save_path=/path/to/folder"""

    parser = argparse.ArgumentParser()

    # base params
    parser.add_argument('--target', type=str)
    parser.add_argument('--save_dir', type=str)

    namespace, _ = parser.parse_known_args()
    run_main(namespace)
