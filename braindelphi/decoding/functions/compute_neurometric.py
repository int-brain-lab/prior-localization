trialsdf_neurometric = nb_trialsdf.reset_index() if (pseudo_id == -1) else \
    pseudosess[pseudomask].reset_index()
if kwargs['model'] is not None:
    blockprob_neurometric = dut.compute_target(
        'pLeft',
        subject,
        subjeids,
        eid,
        kwargs['modelfit_path'],
        binarization_value=kwargs['binarization_value'],
        modeltype=kwargs['model'],
        beh_data_test=trialsdf if pseudo_id == -1 else pseudosess,
        behavior_data_train=behavior_data_train,
        one=one)

    trialsdf_neurometric['blockprob_neurometric'] = np.stack([
        np.greater_equal(
            blockprob_neurometric[(mask & (trialsdf.choice != 0) if pseudo_id
                                                                    == -1 else pseudomask)], border).astype(int)
        for border in kwargs['border_quantiles_neurometric']
    ]).sum(0)

else:
    blockprob_neurometric = trialsdf_neurometric['probabilityLeft'].replace(
        0.2, 0).replace(0.8, 1)
    trialsdf_neurometric['blockprob_neurometric'] = blockprob_neurometric
