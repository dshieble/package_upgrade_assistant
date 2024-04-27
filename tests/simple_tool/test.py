import unittest
from sklearn.linear_model import LinearRegression
from model_training import fit_normalized_model
import numpy as np

class TestMain(unittest.TestCase):

  def test_fit_normalized_model(self):

    X = np.array([[1,2], [2,4], [3,7]])
    y = np.array([1,2,10])

    model = fit_normalized_model(X=X, y=y)
    prediction = model.predict([[5, 12]])
    self.assertAlmostEqual(prediction[0], 16.30375847)

if __name__ == "__main__":
  unittest.main()
