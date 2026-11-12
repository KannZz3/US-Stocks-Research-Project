def c5_model_fit(X, y):
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), verbose=False)
    return model
