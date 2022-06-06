from datetime import date
import numpy as np
import os
import pytest

from braindelphi.decoding.functions.utils import check_settings
from braindelphi.decoding.functions.utils import compute_mask
from braindelphi.decoding.functions.utils import get_save_path


def test_compute_mask(data_dict):

    # TODO: create synthetic trialsdf to have more control

    # allow all trials to come through
    mask = compute_mask(
        data_dict['trialsdf'], align_time='goCue_times', time_window=(-0.0001, 0.0),
        no_unbias=False, min_rt=-10.0, min_len=0.0, max_len=1000.0)
    assert np.sum(mask) == 30

    # get rid of unbiased trials
    mask = compute_mask(
        data_dict['trialsdf'], align_time='goCue_times', time_window=(-0.0001, 0.0),
        no_unbias=True, min_rt=-10.0, min_len=0.0, max_len=1000.0)
    assert np.sum(mask) == 20

    # get rid of short trials
    mask = compute_mask(
        data_dict['trialsdf'], align_time='goCue_times', time_window=(-0.0001, 0.0),
        no_unbias=False, min_rt=0.0, min_len=0.0, max_len=1000.0)
    assert np.sum(mask) == 27

    # get rid of trials where animal does not respond
    trialsdf = data_dict['trialsdf'].copy()
    trialsdf.at[1, 'choice'] = 0
    trialsdf.at[2, 'choice'] = 0
    mask = compute_mask(
        trialsdf, align_time='goCue_times', time_window=(-0.0001, 0.0), no_unbias=False,
        min_rt=-10.0, min_len=0.0, max_len=1000.0)
    assert np.sum(mask) == 28

    # get rid of trials with too fast reaction times
    trialsdf = data_dict['trialsdf'].copy()
    trialsdf.at[1, 'firstMovement_times'] = 0.0
    trialsdf.at[1, 'goCue_times'] = 15.0
    trialsdf.at[2, 'firstMovement_times'] = 0.0
    trialsdf.at[2, 'goCue_times'] = 15.0
    trialsdf.at[3, 'firstMovement_times'] = 0.0
    trialsdf.at[3, 'goCue_times'] = 15.0
    trialsdf.at[4, 'firstMovement_times'] = 0.0
    trialsdf.at[4, 'goCue_times'] = 15.0
    if 'react_times' in trialsdf.columns:
        trialsdf = trialsdf.drop(columns='react_times')  # remove previously computed reaction time
    mask = compute_mask(
        trialsdf, align_time='goCue_times', time_window=(-0.0001, 0.0), no_unbias=False,
        min_rt=-10.0, min_len=0.0, max_len=1000.0)
    assert np.sum(mask) == 26


def test_get_save_path():

    pseudo_id = 10
    subject = 'MRW_000'
    eid = '1234'
    neural_dtype = 'ephys'
    probe = 'probe00'
    region = 'VISpm'
    output_path = os.path.join('test', 'path')
    time_window = (0.1, 1.4)
    today = str(date.today())
    target = 'pLeft'
    add_to_saving_path = 'this_is_a_test'

    outpath = get_save_path(
        pseudo_id=pseudo_id, subject=subject, eid=eid, neural_dtype=neural_dtype, probe=probe,
        region=region, output_path=output_path, time_window=time_window, today=today,
        target=target, add_to_saving_path=add_to_saving_path)

    outpath = str(outpath)
    print(outpath)

    assert outpath.find(str(pseudo_id)) > -1
    assert outpath.find(subject) > -1
    assert outpath.find(eid) > -1
    assert outpath.find(probe) > -1
    assert outpath.find(region) > -1
    assert outpath.find(output_path) > -1
    assert outpath.find(today) > -1
    assert outpath.find(target) > -1
    assert outpath.find(add_to_saving_path) > -1
    assert outpath.endswith('.pkl')


def test_check_setting():

    settings = {'target': 'bad_target_str'}
    with pytest.raises(NotImplementedError):
        check_settings(settings)

    settings = {'target': 'prior', 'align_time': 'bad_align_str'}
    with pytest.raises(NotImplementedError):
        check_settings(settings)
