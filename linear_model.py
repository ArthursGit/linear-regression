import torch
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import numpy as np

class LinearModel:

    def __init__(self, n_features):
        self.theta = torch.tensor(
            np.random.randn(n_features, 1)
        )

    def predict(self, data):
        data = torch.tensor(data)
        assert data.shape[1] == self.theta.shape[0], \
            f"Shape mismatch, expected {self.theta.shape[0]}, got {data.shape[1]}"
        return data @ self.theta


def get_synthetic_data(n_features=1, n_informative=1, noise=5, coefficients=True, random_state=42):
    X, y, coef = ds.make_regression(n_features=n_features,
                              n_informative=n_informative,
                              noise=noise,
                              coef=coefficients,
                              random_state=random_state)

    X = np.c_[np.ones((len(X), 1)), X]

    return X, y, coef


training_data, target, coefficients = get_synthetic_data()


model = LinearModel(2)

prediction = model.predict(training_data)

print(prediction)


