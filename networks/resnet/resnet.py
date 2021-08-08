import numpy as np
from networks.std.network import Model

class ResNet(Model):

	def __init__(self, layers, name='ResNet'):
		self.L = len(layers)
		self.layers = layers
		self.name = name

	def compile(self, error_func, optimizer):
		for l in range(self.L-2):
			self.layers[l].log_next(self.layers[l+1].N, self.layers[l+2].N)
		self.layers[-2].log_next(self.layers[-1].N, 0)
		self.error_func = error_func
		self.optimizer = optimizer
		self.optimizer.compile([layer.N for layer in self.layers])

	def _predict(self, x):
		self.layers[0].feed(x)
		for l in range(1,self.L):
			second_layer = self.layers[l-2] if 0 <= l-2 < self.L-2 else None
			self.layers[l].feed(self.layers[l-1], second_layer)
		return self.layers[-1].neurons

	def predict(self, X):
		return np.array([self._predict(x) for x in X], dtype=np.float32)

