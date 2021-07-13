import numpy as np
from networks.activations import *

class Layer:

	def __init__(self, N, activation=linear, k=1, name='layer', _type=1):
		self.N = N
		self.k = k
		self.neurons = np.zeros(shape=(N,), dtype=np.float32)
		self.thresholds = np.zeros(shape=(N,), dtype=np.float32)
		self.name = name
		self._type = _type
		if isinstance(activation, str):
			global ACTIVATIONS
			self.g = ACTIVATIONS[activation]
		else:
			self.g = activation

	def log_next(self, M):
		stddev = np.sqrt(self.k/float(self.N))
		self.weights = np.random.normal(scale=stddev, size=(M,self.N)) # mean is 0 by default

	def get_local_fields(self, bias):
		return np.dot(self.weights, self.neurons)+bias

	# feed a previous layer or an input
	def feed(self, prev):
		if not isinstance(prev, np.ndarray) and not isinstance(prev, Layer):
			raise TypeError("Expected numpy array or layer object but received {type} instead".format(type=type(prev)))
		if isinstance(prev, Layer):
			self.neurons = self.g(prev.get_local_fields(self.thresholds))
		elif isinstance(prev, np.ndarray):
			self.neurons = prev

	def get_trainable_params(self):
		return (0 if 'weights' not in dir(self) else self.weights.shape[0]*self.weights.shape[1])+len(self.thresholds)

	# stores values in an h5py file so that they can easily be saved
	def save(self, file, layer_idx):
		global ACTIVATIONS
		group = file.create_group(f'layer_{layer_idx}')
		if 'weights' in dir(self):
			group.create_dataset('weights', self.weights.shape, np.float32, self.weights, compression="gzip")
		group.create_dataset('thresholds', self.thresholds.shape, np.float32, self.thresholds, compression="gzip")
		name = np.array([ord(x) for x in self.name], dtype=np.ubyte)
		group.create_dataset('name', name.shape, np.ubyte, name, compression="gzip")
		activation_index = list(ACTIVATIONS.values()).index(self.g)
		activation_index = np.array(activation_index, dtype=np.ubyte)
		group.create_dataset('activation', (1,), np.ubyte, activation_index, compression="gzip")
		group.create_dataset('type', (1,), np.ubyte, [self._type], compression="gzip")
		group.create_dataset('k', (1,), np.float32, [self.k], compression="gzip")

	def load(self, file, layer_idx):
		global ACTIVATIONS
		group = file[f'layer_{layer_idx}']
		try:
			self.weights = np.array(group['weights'], dtype=np.float32)
		except KeyError:
			pass
		self.thresholds = np.array(group['thresholds'], dtype=np.float32)
		self.N = len(self.thresholds)
		self.neurons = np.zeros(shape=(self.N,), dtype=np.float32)
		self.name = ''.join([chr(x) for x in group['name']])
		self.g = list(ACTIVATIONS.values())[group['activation'][0]]
		self._type = group['type'][0]
		self.k = group['k'][0]

	def __str__(self):
		return "-"*40+"\nDense [neurons:{neurons}, k={k}, activation={g}]".format(neurons=self.N, k=self.k, g=self.g.__name__)+"\n"

	def __call__(self, x):
		self.feed(x)

class ELU(Layer):

	def __init__(self, N, activation=ELU, k=1, alpha=0.1, name='ELU layer'):
		super().__init__(N, activation, k=k, _type=2)
		self.alpha = alpha

	def get_activation(self, thresholds):
		return self.g(self.get_local_fields(thresholds), deriv=False, alpha=self.alpha)

	def __str__(self):
		return "-"*40+"\nELU [neurons:{neurons}, alpha={alpha}, k={k}]".format(neurons=self.N,alpha=self.alpha,k=self.k)+"\n"