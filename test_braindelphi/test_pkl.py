#!/usr/bin/env python
# @File: test_braindelphi/test_pkl.py
# @Author: Niccolo' Bonacchi (@nbonacchi)
# @Date: Sunday, May 22nd 2022, 3:29:12 pm
from pathlib import Path

import iblutil.util
import numpy as np
import pandas as pd
from braindelphi.utils_root import load_pickle_data


def test_load_pickle_data():
    pkl_path = Path(__file__).parent.joinpath("test_data.pkl")
    data = load_pickle_data(pkl_path)
    data_keys = ["trials_df", "spk_times", "spk_clu", "clu_regions", "clu_qc", "clu_df"]
    assert type(data) is dict, "Did not load dict"
    assert [x for x in data.keys()] == data_keys, "Did not load all keys"
    assert type(data["trials_df"]) is pd.DataFrame, "trials_df should be a DataFrame"
    assert type(data["spk_times"]) is np.ndarray, "spk_times should be a NumPy array"
    assert type(data["spk_clu"]) is np.ndarray, "spk_clu should be a NumPy array"
    assert type(data["clu_regions"]) is np.ndarray, "clu_regions should be a DataFrame"
    assert type(data["clu_qc"]) is iblutil.util.Bunch, "clu_qc should be a Bunch"
    assert type(data["clu_df"]) is pd.DataFrame, "clu_df should be a DataFrame"
