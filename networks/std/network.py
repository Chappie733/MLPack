import numpy as np
from networks.std.layer import Layer

class Model:

	def __init__(self, layers):
		self.L = len(layers)
		self.layers = layers

	def compile(self, error_func, optimizer):
		for l in range(self.L-1):
			self.layers[l].log_next(self.layers[l+1].N)
		self.error_func = error_func
		self.optimizer = optimizer

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

	def fit(self, inputs, labels, epochs=75, verbose=True):
		for epoch in range(1, epochs+1):
			if verbose:
				H = 0

			for u in range(len(inputs)):
				X, Y = inputs[u], labels[u]

				if verbose:
					predicted = self._predict(X)
					H += self.error_func(Y, predicted)

				B_L = self.layers[-2].get_local_fields(self.layers[-1].thresholds)
				errors_L = self.error_func.grad(Y, predicted)*self.layers[-2].g(B_L, deriv=True)
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
					updates = self.optimizer.step(gradients, layer=l-1)
					self.layers[l-1].weights += np.reshape(updates[:-self.layers[l].N], (self.layers[l].N, self.layers[l-1].N))
					self.layers[l].thresholds += updates[-self.layers[l].N:]

			print("Error on epoch #{epoch}: {H}".format(epoch=epoch, H=H))
