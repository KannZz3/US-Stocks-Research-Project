def c5_model_fit(X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = ElasticNet(**ENET_PARAMS)
        model.fit(np.asfortranarray(X, dtype=float), np.asarray(y, dtype=float))
    if any((issubclass(w.category, ConvergenceWarning) for w in caught)):
        raise RuntimeError('ElasticNet convergence warning: inspect alpha / iterations.')
    return model
