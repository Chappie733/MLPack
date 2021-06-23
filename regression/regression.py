import numpy as np
from numbers import Number


class Regressor:

	# args = [learning rate, weights standard deviation]
	def __init__(self, N, lr, stddev):
		self.N = N
		self.lr = lr
		self.weights = np.random.normal(scale=stddev, size=(N,))
		self.theta = 0
		self.func = None

	def B(self, X):
		return np.dot(X, self.weights)+self.theta

	def predict(self, X):
		return self.func(self.B(X))

	def train(self, X, Y, epochs=50, verbose=False):
		if not isinstance(Y, np.ndarray):
			Y = np.array(Y)
		if not isinstance(X, np.ndarray):
			X = np.array(X)

		for epoch in range(1, epochs+1):
			B = self.B(X)
			if verbose:
				print("Epoch #{epoch}: {H}".format(epoch=epoch, H=self.H(Y, self.func(B))))

			dw, dt = self.step(X, Y, B)

			self.weights = self.weights + dw
			self.theta = self.theta + dt

class LinearRegressor(Regressor):

	def __init__(self, N, lr=0.1, stddev=1):
		super().__init__(N, lr=lr, stddev=stddev)
		self.func = lambda x: x

	def H(self, T, Y):
		return np.sum((T-Y)**2)/2

	def step(self, X, Y, B):
		dw = np.dot((Y-B), X)*self.lr
		dt = np.sum(Y-B)*self.lr
		return dw, dt

class LogisticRegressor(Regressor):

	def __init__(self, N, lr=0.1, stddev=1):
		super().__init__(N, lr=lr, stddev=stddev)

	def func(self, x, deriv=False):
		return 1/(1+np.exp(-x)) if not deriv else self.func(x)*(1-self.func(x))

	def H(self, T, Y):
		return -np.sum(T*np.log(Y)+(1-T)*np.log(1-Y))

	def step(self, X, Y, B):
		predictions = self.func(B)
		cmt = (Y-predictions)/(predictions*(1-predictions))*self.func(B, deriv=True) # common term

		dw = np.dot(cmt, X)*self.lr
		dt = np.sum(cmt)*self.lr

		return dw, dt