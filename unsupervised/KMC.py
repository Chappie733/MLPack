import numpy as np
import h5py
import os

# K-Means Clustering
class KMC:

	# N -> amount of features
	# K -> amount of clusters
	def __init__(self, N, K, lr=0.01):
		self.N = N
		self.K = K
		self.weights = np.random.normal(scale=1, size=(K,N))
		self.lr = lr

	def _classify(self, x):
		return np.argmin(np.sum((self.weights-x)**2, axis=1))

	def classify(self, X):
		return np.array([self._classify(x) for x in X])

	def fit_mean(self, X, epochs):
		for epoch in range(1, epochs+1):
			for i in range(self.K):
				indices = np.where(self.classify(X)==i)[0]
				self.weights[i] = np.mean(X[indices])

	def fit_descent(self, X, epochs):
		for epoch in range(1, epochs+1):
			for i in range(self.K):
				indices = np.where(self.classify(X)==i)[0]
				dw = np.zeros(self.N)
				for u in indices:
					dw = dw + np.linalg.norm(X[u]-self.weights[i])*(X[u]-self.weights[i])
				self.weights[i] = self.weights[i] + self.lr*dw

	def fit(self, X, epochs=1000, mode='mean'):
		if mode == 'mean':
			self.fit_mean(X, epochs)
		elif mode == 'descent':
			self.fit_descent(X, epochs)


	def __str__(self):
		return "-"*20+f"K-Means Clusterer:\nClusters: {self.K}\nFeatures: {self.N}\nLearning rate: {self.lr}"+"-"*20

	def save(self, filename, absolute=False):
		path = filename if absolute else os.path.join(os.getcwd(), filename)
		file = h5py.File(path+'.h5', 'w')
		file.create_dataset('weights', self.weights.shape, np.float32, self.weights, compression='gzip')
		file.create_dataset('lr', (1,), np.float64, [self.lr], compression='gzip')
		file.close()

	def load(self, filename, absolute=False):
		path = filename if absolute else os.path.join(os.getcwd(), filename)
		file = h5py.File(path + '.h5', 'r')
		self.weights = np.array(file['weights'], dtype=np.float32)
		self.lr = file['lr'][0]
		self.K, self.N = self.weights.shape
		file.close()