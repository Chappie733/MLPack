import numpy as np
import networks.std.layer as Layer

class ResLayer(Layer.Layer):

	def __init__(self, N, activation='linear', k=1, name='Residual Layer'):
		super().__init__(N, activation=activation, k=k, name=name, _type=4)

	def log_next(self, M, P=0):
		stddev = np.sqrt(self.k/float(self.N))
		self.weights = np.random.normal(scale=stddev, size=(M, self.N)) # feeding into the next layer
		if P != 0:
			self.weights_res = np.random.normal(scale=stddev, size=(P, self.N))

	# previous is whether the connection is to the successive  
	# layer (l+1) or to the one after that (l+2)
	def get_local_fields(self, bias, previous=True):
		return np.dot(self.weights if previous else self.weights_res, self.neurons)+bias

	def feed(self, x1, x2=None):
		if not isinstance(x1, np.ndarray) and not isinstance(x1, Layer.Layer):
			raise TypeError(f"Expected numpy array or layer object but received {type(x1)} instead")
		if not isinstance(x2, np.ndarray) and not isinstance(x2, Layer.Layer):
			if x2 is not None:
				raise TypeError(f"Expected numpy array or layer object but received {type(x2)} instead")
		if isinstance(x1, Layer.Layer):
			residual_local_fields = 0 if x2 is None else x2.get_local_fields(0, previous=False)
			self.neurons = self.g(x1.get_local_fields(self.thresholds, previous=True)+residual_local_fields)
		elif isinstance(x1, np.ndarray):
			self.neurons = x1

	def save(self, file, layer_idx):
		super().save(file, layer_idx)
		group = file[f'layer_{layer_idx}']
		if 'weights_res' in dir(self):
			group.create_dataset('weights_res', self.weights_res.shape, np.float32, self.weights_res, compression="gzip")

	def load(self, file, layer_idx):
		super().load(file, layer_idx)
		group = file[f'layer_{layer_idx}']
		if 'weights_res' in group.keys():
			self.weights_res = np.array(group['weights_res'], dtype=np.float32)

	def get_trainable_params(self):
		return super().get_trainable_params()+(0 if 'weights_res' not in dir(self) else self.weights_res.shape[0]*self.weights_res.shape[1])

	def back_transform(self, errors):
		try:
			return np.dot(self.weights.T, errors)+np.dot(self.weights_res, errors)
		except AttributeError:
			return np.dot(self.weights.T, errors)