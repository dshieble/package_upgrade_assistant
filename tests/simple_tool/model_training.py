from sklearn.linear_model import Ridge

def fit_normalized_model(X, y):
  model = Ridge(normalize=True)
  model.fit(X, y)
  return model

