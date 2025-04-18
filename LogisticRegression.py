import numpy as np


class LogisticRegression:
	def __init__(self, max_iter):
		self.max_iter = max_iter
		self.theta = None

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.theta = np.zeros(n_features)
		for _ in range(self.max_iter):
			z = np.dot(X, self.theta)
			h = self.sigmoid(z)
			gradient = np.dot(X.T, (h - y)) / n_samples
			self.theta -= 0.01 * gradient

	def predict_proba(self, X):
		z = np.dot(X, self.theta)
		return self.sigmoid(z)

	def predict(self, X):
		proba = self.predict_proba(X)
		return (proba >= 0.5).astype(int)
