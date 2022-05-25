import pytest

from braindelphi.decoding.functions.process_inputs import build_predictor_matrix


def test_build_predictor_matrix():

    n_t = 100
    n_clusters = 7
    array = np.random.randn(n_t, n_clusters)

    # no lags
    n_lags = 0
    mat = build_predictor_matrix(array, n_lags, return_valid=True)
    assert np.allclose(mat, array)

    # invalid lags
    n_lags = -1
    with pytest.raises(ValueError):
        mat = build_predictor_matrix(array, n_lags, return_valid=True)

    # positive lags, with and without valid returns
    for n_lags in [1, 2, 3]:
        mat = build_predictor_matrix(array, n_lags, return_valid=False)
        assert mat.shape == (n_t, n_clusters * (n_lags + 1))
        assert np.allclose(array, mat[:, :n_clusters])
        mat = build_predictor_matrix(array, n_lags, return_valid=True)
        assert mat.shape == (n_t - n_lags, n_clusters * (n_lags + 1))
        assert np.allclose(array[n_lags:], mat[:, :n_clusters])
