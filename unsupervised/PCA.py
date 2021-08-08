import numpy as np
import h5py
import os

# TODO: implement saving and loading
class PCA:

	# N is the amount of features, M is the amount of clusters to form
	def __init__(self, N, M, lr=0.01, var=0.1, tau_lr=2, tau_var=2, name='PCA'):
		self.N = N
		self.M = M
		self.lr_start = lr
		self.var_start = var
		self.tau_lr = tau_lr
		self.tau_var = tau_var
		self.name = name

	def _classify(self, x):
		return np.argmin(np.sum((self.weights-x)**2, axis=1))

	def classify(self, X):
		return np.array([self._classify(x) for x in X])

	def fit(self, X, epochs=1000, var_decay=True):
		self.weights = np.random.normal(scale=1, size=(self.M, self.N))

		for epoch in range(epochs):
			k = np.random.randint(low=0, high=len(X))
			wci = self._classify(X[k]) # winning neuron index
			lr = self.lr_start*np.exp(-epoch/self.tau_lr) if var_decay else self.lr_start
			var = self.var_start*np.exp(-epoch/self.tau_var) if var_decay else self.var_start
			for i in range(self.M):
				neighborhood_func = np.exp(-np.linalg.norm(self.weights[i]-self.weights[wci])/(2*var))
				self.weights[i] = self.weights[i] + lr*neighborhood_func*(X[k]-self.weights[i])

	def __str__(self):
		return "-"*20+f"\nPrincipal component analysis\nFeatures: {self.N}\nClasses: {self.M}"

	def save(self, filename, absolute=False):
		path = filename if absolute else os.path.join(os.getcwd(), filename)
		file = h5py.File(path+'.h5', 'w')
		file.create_dataset('weights', self.weights.shape, np.float32, self.weights, compression="gzip")
		file.create_dataset('params', (4,), np.float64, [self.lr_start, self.var_start, self.tau_lr, self.tau_var], compression="gzip")
		name_ascii = np.array([ord(x) for x in self.name], dtype=np.ubyte)
		file.create_dataset('name', name_ascii.shape, np.ubyte, name_ascii, compression="gzip")
		file.close()

	def load(self, filename, absolute=False):
		path = filename if absolute else os.path.join(os.getcwd(), filename)
		file = h5py.File(path+'.h5', 'r')
		self.weights = np.array(file['weights'])
		self.lr_start, self.var_start, self.tau_lr, self.tau_var = file['params']
		self.name = ''.join([chr(x) for x in file['name']])
		self.M, self.N = self.weights.shape
		file.close()