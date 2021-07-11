import numpy as np
import os
import h5py

class Model:

	def __init__(self, layers, name='model'):
		self.L = len(layers)
		self.layers = layers
		self.name = name

	def compile(self, error_func, optimizer):
		for l in range(self.L-1):
			self.layers[l].log_next(self.layers[l+1].N)
		self.error_func = error_func
		self.optimizer = optimizer
		self.optimizer.compile([layer.N for layer in self.layers])

	def _predict(self, x):
		self.layers[0].feed(x)
		for l in range(1, self.L):
			self.layers[l].feed(self.layers[l-1])
		return self.layers[-1].neurons

	def predict(self, X):
		return np.array([self._predict(x) for x in X], dtype=np.float32)

	def __call__(self, X):
		return self.predict(X)

	# returns the value of the error function
	def evaluate(self, X, Y):
		if not isinstance(Y, np.ndarray):
			Y = np.array(Y)
		predictions = self.predict(X)
		return self.error_func(Y, predictions)

	def fit_stochastic(self, inputs, labels, epochs=75, verbose=True, return_errors=False):
		if return_errors:
			vals = []

		for epoch in range(1, epochs+1):
			if verbose or return_errors:
				H = 0

			for u in range(len(inputs)):
				X, Y = inputs[u], labels[u]
				predicted = self._predict(X)

				if verbose or return_errors:
					H += self.error_func(Y, predicted)

				B_L = self.layers[-2].get_local_fields(self.layers[-1].thresholds)
				errors_L = self.error_func.grad(Y, predicted)*self.layers[-1].g(B_L, deriv=True)
				errors = [errors_L]

				for l in range(self.L-1, 1, -1):
					B_l = self.layers[l-2].get_local_fields(self.layers[l-1].thresholds)
					g_prime = self.layers[l-1].g(B_l, deriv=True)

					error_l = np.dot(self.layers[l-1].weights.T, errors[-1])
					errors.append(error_l*g_prime)

				for l in range(1, self.L):
					weights_grads = np.zeros(self.layers[l-1].weights.shape)
					thresholds_grads = errors[-l]
					for j in range(self.layers[l].N):
						for i in range(self.layers[l-1].N):
							weights_grads[j][i] = errors[-l][j]*self.layers[l-1].neurons[i]
					
					gradients = np.append(weights_grads, thresholds_grads)
					updates = self.optimizer.step(gradients, layer=l-1, epoch=epoch)
					self.layers[l-1].weights += np.reshape(updates[:-self.layers[l].N], (self.layers[l].N, self.layers[l-1].N))
					self.layers[l].thresholds += updates[-self.layers[l].N:]

			if verbose:
				print("Error on epoch #{epoch}: {H}".format(epoch=epoch, H=H))
			if return_errors:
				vals.append(H)
		if return_errors:
			return vals

	def fit(self, inputs, labels, epochs=75, batches=True, batch_size=32, verbose=True, return_errors=False):
		if return_errors:
			vals = []

		if not batches or batch_size == 1:
			return self.fit_stochastic(inputs, labels, epochs, verbose, return_errors)

		for epoch in range(1, epochs+1):

			if verbose or return_errors:
				H = 0

			for batch in range(int(np.ceil(len(labels)/batch_size))):
				updates = [np.zeros(shape=(self.layers[i].N*(self.layers[i-1].N+1),)) for i in range(1,self.L)]
				for u in range(batch*batch_size, min((batch+1)*batch_size, len(inputs))):
					X, Y = inputs[u], labels[u]
					predicted = self._predict(X)
					
					if verbose or return_errors:
						H += self.error_func(Y, predicted)

					B_L = self.layers[-2].get_local_fields(self.layers[-1].thresholds)
					errors_L = self.error_func.grad(Y, predicted)*self.layers[-1].g(B_L, deriv=True)
					errors = [errors_L]

					for l in range(self.L-1, 1, -1):
						B_l = self.layers[l-2].get_local_fields(self.layers[l-1].thresholds)
						g_prime = self.layers[l-1].g(B_l, deriv=True)

						error_l = np.dot(self.layers[l-1].weights.T, errors[-1])
						errors.append(error_l*g_prime)

					for l in range(1, self.L):
						weights_grads = np.zeros(self.layers[l-1].weights.shape)
						thresholds_grads = errors[-l]
						for j in range(self.layers[l].N):
							for i in range(self.layers[l-1].N):
								weights_grads[j][i] = errors[-l][j]*self.layers[l-1].neurons[i]
						
						gradients = np.append(weights_grads, thresholds_grads)
						updates[l-1] += self.optimizer.step(gradients, layer=l-1, epoch=epoch)
					
				for l in range(1, self.L):
					self.layers[l-1].weights += np.reshape(updates[l-1][:-self.layers[l].N], (self.layers[l].N, self.layers[l-1].N))/batch_size
					self.layers[l].thresholds += updates[l-1][-self.layers[l].N:]/batch_size

			if verbose:
				print("Error on epoch #{epoch}: {H}".format(epoch=epoch, H=H))
			if return_errors:
				vals.append(H)
		if return_errors:
			return vals	

	def __str__(self):
		res = f"Sequential Model: name: {self.name}, layers: {self.L}\n"
		trainable_params = 0
		for layer in self.layers:
			res += str(layer)
			trainable_params += layer.get_trainable_params()
		res += f"Trainable parameters: {trainable_params}"
		return res

	def save(self, filename, absolute=False):
		dirname = os.getcwd()
		path = filename if absolute else os.path.join(dirname, filename)
		file = h5py.File(path, 'w')
		structure = np.array([[layer.N, layer._type] for layer in self.layers], dtype=np.uint32)
		for i in range(self.L):
			self.layers[i].save(file, i)

		file.create_dataset('structure', structure.shape, np.uint32, structure, compression="gzip")
		name = np.array([ord(x) for x in self.name], dtype=np.ubyte)
		file.create_dataset('name', name.shape, np.ubyte, name, compression='gzip')
		file.close()