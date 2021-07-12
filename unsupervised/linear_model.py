import numpy as np
import h5py
import os

# classifies the inputs vectors X (of size N) in M different classes based on their features
class LinearModel:

	def __init__(self, N, M, lr=0.01):
		self.weights = np.random.normal(scale=1, size=(M, N))
		self.lr = lr
		self.N = N
		self.M = M

	def fit(self, X, epochs=100):
		if X.shape[1] != self.N:
			raise IndexError(f"Expected the input vectors to have size {self.N} but received vectors with size {X.shape[1]}!")

		for epoch in range(1, epochs+1):
			classifications = [np.argmax(np.dot(self.weights, x)) for x in X]

			for idx, win_idx in enumerate(classifications):
				self.weights[win_idx] = self.weights[win_idx]+self.lr*(X[idx]-self.weights[win_idx])

	def _classify(self, x):
		return np.argmax(np.dot(self.weights, x))

	def classify(self, X):
		return np.array([self._classify(x) for x in X])

	def __call__(self, X):
		return self.classify(X)

	def __str__(self):
		return "-"*20+f"Unsupervised Linear Model:\nfeatures: {self.N}\nclasses: {self.M}\nlearning rate: {self.lr}\n" + "-"*20

	def get_cluster_position(self, class_idx):
		return self.weights[class_idx]

	# absolute is whether the path is absolute or relative
	def save(self, filename, absolute=False):
		path = filename if not absolute else os.path.join(os.getcwd(), filename)
		file = h5py.File(path+'.h5', 'w')
		file.create_dataset('weights', self.weights.shape, np.float32, self.weights, compression='gzip')
		file.create_dataset('lr', (1,), np.float64, [self.lr], compression='gzip')
		file.close()

	def load(self, filename, absolute=False):
		path = filename if not absolute else os.path.join(os.getcwd(), filename)
		file = h5py.File(path + '.h5', 'r')
		self.weights = np.array(file['weights'], dtype=np.float32)
		self.lr = file['lr'][0]
		file.close()