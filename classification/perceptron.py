import numpy as np

class Perceptron:

	def __init__(self, N, lr=0.1):
		self.weights = np.zeros(N)
		self.bias = 0
		self.N = N
		self.lr = lr # only used for Rosenblatt rule

	def predict(self, x):
		return np.sign(np.dot(x, self.weights)+self.bias)

	# Pocket learning algorithm
	def pocket(self, X, Y, steps):
		pi = np.zeros(self.N)
		pi_bias = 0
		num_ok_pi, num_ok_w = 0, 0
		streak_pi, streak_w = 0, 0

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
							training_complete = True
							break
			else:
				streak_pi = 0
				pi = pi + Y[k]*X[k]
				pi_bias = pi_bias + Y[k]

	def pla(self, X, Y, steps):
		for step in range(steps):
			k = np.random.randint(len(Y))
			if self.predict(X[k]) != Y[k]:
				self.weights = self.weights + Y[k]*X[k]
				self.bias = self.bias + Y[k]

	def Rosenblatt(self, X, Y, steps):
		for step in range(steps):
			k = np.random.randint(len(Y))
			prediction = self.predict(X[k])
			if prediction != Y[k]:
				self.weights = self.weights + self.lr*(Y[k]-prediction)*X[k]
				self.bias = self.bias + self.lr*Y[k]


	def fit(self, X, Y, method='pocket', **kwargs):
		steps = 1000 if 'steps' not in kwargs else kwargs['steps']
		if method == 'pocket':
			self.pocket(X, Y, steps)
		elif method == 'perceptron_learning_algorithm' or method == 'pla':
			self.pla(X,Y, steps)
		elif method == 'rosenblatt':
			self.Rosenblatt(X,Y,steps)
		else:
			print(f"Method {method} not implemented!")