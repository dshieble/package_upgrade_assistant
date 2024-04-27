from sklearn.linear_model import Ridge
import numpy as np

def fit_normalized_model(X, y):
  X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
  model = Ridge()
  model.fit(X_normalized, y)
  return model

