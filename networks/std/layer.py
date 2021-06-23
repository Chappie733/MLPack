import numpy as np
from networks.activations import *

class Layer:

	def __init__(self, N, activation=linear, k=1):
		self.N = N
		self.g = activation
		self.k = k
		self.neurons = np.zeros(shape=(N,), dtype=np.float32)
		self.thresholds = np.zeros(shape=(N,), dtype=np.float32)

	def log_next(self, M):
		stddev = np.sqrt(self.k/float(self.N))
		self.weights = np.random.normal(scale=stddev, size=(M,self.N)) # mean is 0 by default

	def get_local_fields(self, bias):
		return np.dot(self.weights, self.neurons)+bias

	def get_activation(self, thresholds):
		return self.g(self.get_local_fields(thresholds))

	# feed a previous layer or an input
	def feed(self, prev):
		if not isinstance(prev, np.ndarray) and not isinstance(prev, Layer):
			raise TypeError("Expected numpy array or layer object but received {type} instead".format(type=type(prev)))
		if isinstance(prev, Layer):
			self.neurons = prev.get_activation(self.thresholds)
		elif isinstance(prev, np.ndarray):
			self.neurons = prev

	def __str__(self):
		return "-"*10+"\nDense [neurons:{neurons}, k={k}, activation={g}]".format(neurons=self.N, k=self.k, g=self.g.__name__)+"\n"

	def __call__(self, x):
		self.feed(x)

class ELU(Layer):

	def __init__(self, N, activation=ELU, k=1, alpha=0.1):
		super().__init__(N, activation, k)
		self.alpha = alpha

	def get_activation(self, thresholds):
		return self.g(self.get_local_fields(thresholds), deriv=False, alpha=self.alpha)

	def __str__(self):
		return "-"*10+"\nELU [neurons:{neurons}, alpha={alpha}, k={k}]".format(neurons=self.N,alpha=self.alpha,k=self.k)+"\n"