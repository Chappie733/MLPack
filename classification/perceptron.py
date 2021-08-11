import numpy as np
from classification.classifier import Classifier

class Perceptron(Classifier):

	def __init__(self, N, lr=0.1, name="Perceptron"):
		super().__init__(N, 2, name)
		self.weights = np.zeros(N)
		self.bias = 0
		self.lr = lr # only used for Rosenblatt rule

	def predict(self, x):
		return (np.sign(np.dot(x, self.weights)+self.bias)+1)/2

	# Pocket learning algorithm
	def _pocket(self, X, Y, steps):
		pi = np.zeros(self.N)
		pi_bias = 0
		streak_pi, streak_w = 0, 0
		num_ok_pi, num_ok_w = 0, 0

		for step in range(steps):
			k = np.random.randint(len(Y))
			if np.sign(np.dot(X[k], pi)+pi_bias) == Y[k]:
				streak_pi += 1
				
				if streak_pi > streak_w:
					num_ok_pi = len(np.where(np.sign(np.dot(X, pi)+pi_bias) == Y)[0])
					if num_ok_pi > num_ok_w:
						streak_w = streak_pi
						num_ok_w = num_ok_pi
						self.weights = np.copy(pi)
						self.bias = pi_bias
						if num_ok_w == len(Y):
							break
			else:
				streak_pi = 0
				pi = pi + Y[k]*X[k]
				pi_bias = pi_bias + Y[k]

	def _pla(self, X, Y, steps):
		for step in range(steps):
			k = np.random.randint(len(Y))
			if self.predict(X[k]) != Y[k]:
				self.weights = self.weights + Y[k]*X[k]
				self.bias = self.bias + Y[k]

	def _Rosenblatt(self, X, Y, steps):
		for step in range(steps):
			k = np.random.randint(len(Y))
			prediction = self.predict(X[k])
			if prediction != Y[k]:
				self.weights = self.weights + self.lr*(Y[k]-prediction)*X[k]
				self.bias = self.bias + self.lr*(Y[k]-prediction)


	def fit(self, X, Y, method='pocket', *args, **kwargs):
		# if the label data assumes the two classes to be [0,1] instead of [-1,1]
		# this immediately fixes the problem so that the learning can take place
		if np.array_equal(np.unique(Y), [0,1]):
			Y = 2*Y-1

		steps = 1000 if 'steps' not in kwargs else kwargs['steps']
		if method == 'pocket':
			self._pocket(X, Y, steps)
		elif method == 'perceptron_learning_algorithm' or method == 'pla':
			self._pla(X,Y, steps)
		elif method == 'rosenblatt':
			self._Rosenblatt(X,Y,steps)
		else:
			print(f"Method {method} not implemented!")

	def _save(self, file):
		file.create_dataset('weights', self.weights.shape, np.float32, self.weights, compression="gzip")
		file.create_dataset('params', (2,), np.float32, [self.lr, self.bias], compression="gzip")

	def _load(self, file):
		self.weights = np.array(file['weights'])
		self.lr, self.bias = file['params']		