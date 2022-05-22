def pdf_from_histogram(x, out):
    # unit test of pdf_from_histogram
    # out = np.histogram(np.array([0.9, 0.9]), bins=target_distribution[-1], density=True)
    # out[0][(np.array([0.9])[:, None] > out[1][None]).sum(axis=-1) - 1]
    return out[0][(x[:, None] > out[1][None]).sum(axis=-1) - 1]


def test_df_from_histogram(target_distribution):
    v = 0.9
    out = np.histogram(np.array([v, v]), bins=target_distribution[-1], density=True)
    assert (pdf_from_histogram(np.array([v]), out) > 0)


def balanced_weighting(vec, continuous, use_openturns, bin_size_kde, target_distribution):
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.KernelSmoothing.html?highlight=kernel%20smoothing
    # This plug-in method for bandwidth estimation is based on the solve-the-equation rule from (Sheather, Jones, 1991).
    if continuous:
        if use_openturns:
            factory = openturns.KernelSmoothing()
            sample = openturns.Sample(vec[:, None])
            bandwidth = factory.computePluginBandwidth(sample)
            distribution = factory.build(sample, bandwidth)
            proposal_weights = np.array(distribution.computePDF(sample)).squeeze()
            balanced_weight = np.ones(vec.size) / proposal_weights
        else:
            emp_distribution = np.histogram(vec, bins=target_distribution[-1], density=True)
            balanced_weight = pdf_from_histogram(vec, target_distribution) / pdf_from_histogram(
                vec, emp_distribution)
        #  plt.hist(y_train_inner[:, None], density=True)
        #  plt.plot(sample, proposal_weights, '+')
    else:
        balanced_weight = compute_sample_weight("balanced", y=vec)
    return balanced_weight

    if kwargs['balanced_weight'] and kwargs['balanced_continuous_target']:
        if (kwargs['no_unbias'] and not kwargs['use_imposter_session_for_balancing'] and
            (kwargs['model'] == dut.optimal_Bayesian)):
            with open(
                    kwargs['decoding_path'].joinpath(
                        'targetpLeft_optBay_%s.pkl' %
                        str(kwargs['bin_size_kde']).replace('.', '_')), 'rb') as f:
                target_distribution = pickle.load(f)
        elif not kwargs['use_imposter_session_for_balancing'] and (kwargs['model']
                                                                   == dut.optimal_Bayesian):
            target_distribution, _ = dut.get_target_pLeft(nb_trials=trialsdf.index.size,
                                                          nb_sessions=250,
                                                          take_out_unbiased=False,
                                                          bin_size_kde=kwargs['bin_size_kde'])
        else:
            subjModel = {
                'modeltype': kwargs['model'],
                'subjeids': subjeids,
                'subject': subject,
                'modelfit_path': kwargs['modelfit_path'],
                'imposterdf': kwargs['imposterdf'],
                'use_imposter_session_for_balancing': kwargs['use_imposter_session_for_balancing'],
                'eid': eid,
                'target': kwargs['target']
            }
            target_distribution, allV_t = dut.get_target_pLeft(
                nb_trials=trialsdf.index.size,
                nb_sessions=250,
                take_out_unbiased=kwargs['no_unbias'],
                bin_size_kde=kwargs['bin_size_kde'],
                subjModel=subjModel)
    else:
        target_distribution = None
