import numpy as np

class LinearMachine:

	def __init__(self, N, M, lr=0.001, name='linear machine'):
		self.name = name
		self.N = N # elements in the feature vector
		self.M = M # classes
		self.weights = np.zeros((M, N))

	def _predict(self, x):
		return np.argmax(np.dot(self.weights, x))

	def predict(self, X):
		return np.array([self._predict(x) for x in X])

	def fit(self, X, Y, steps=1000):
		pi = np.random.normal(scale=1, size=(self.M, self.N))
		streak_pi, streak_w = 0, 0
		num_ok_w = 0

		for step in range(1, steps+1):
			k = np.random.randint(low=0, high=len(Y))
			i = np.argmax(np.dot(pi, X[k])) # index of the winning neuron
			if i == Y[k]:
				streak_pi += 1
				if streak_pi >= streak_w:
					pi_predictions = np.array([np.argmax(np.dot(pi, x)) for x in X])
					num_ok_pi = len(np.where(pi_predictions == Y)[0])
					if num_ok_pi > num_ok_w:
						streak_w = streak_pi
						num_ok_pi = num_ok_w
						self.weights = pi.copy()
						if num_ok_pi == len(Y):
							break
			else:
				pi[i] = pi[i] - 2*X[k]
				pi = pi + X[k]

	def get_accuracy(self, X, Y):
		predictions = self.predict(X)
		return len(np.where(predictions==Y)[0])/len(Y)