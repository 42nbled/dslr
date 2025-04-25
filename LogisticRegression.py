import numpy as np
from pathlib import Path

class LogisticRegression:
	def __init__(self, max_iter):
		self.max_iter = max_iter
		self.theta = None

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def fit(self, X, y, learning_rate=0.01):
		n_samples, n_features = X.shape
		self.theta = np.zeros(n_features)
		for _ in range(self.max_iter):
			z = np.dot(X, self.theta)
			h = self.sigmoid(z)
			gradient = np.dot(X.T, (h - y)) / n_samples
			self.theta -= learning_rate * gradient

	def predict_proba(self, X):
		if self.theta is None:
			print('No trained model found')
			exit(1)
		z = np.dot(X, self.theta)
		return self.af_softmax(self.sigmoid(z))

	def af_softmax(self, x):
		e_x = np.exp(x -np.max(x, axis=0))
		return e_x / np.sum(e_x, axis=0)

	def save(self, file):
		if self.theta is None:
			print('No trained model found')
			exit(1)
		np.save(file, self.theta)

	def load(self, file):
		self.theta = np.load(file)

