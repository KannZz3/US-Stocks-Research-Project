def c5_model_fit(X, y):
    model = LGBMRegressor(**LGB_PARAMS)
    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
    return model
