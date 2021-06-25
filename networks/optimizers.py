import numpy as npc	

class Optimizer:

	def __init__(self, name, lr):
		self.name = name
		self.lr = lr


class SGD(Optimizer):

	def __init__(self, lr=0.001):
		super().__init__('Standard Gradient Descent', lr)

	def step(self, grads, **kwargs):
		return -self.lr*grads

	def __call__(self, grads, **kwargs):
		return self.step(grads)

class Momentum(Optimizer):

	def __init__(self, lr=0.001, momentum=0.1):
		super().__init__('Momentum', lr)
		self.v_prev = None
		self.momentum = momentum

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		if self.v_prev is None:
			self.v_prev = [np.zeros(grads.shape)]
		elif layer >= len(self.v_prev):
			self.v_prev.append(np.zeros(grads.shape)) 

		v_t = self.momentum*self.v_prev[layer]+self.lr*grads
		self.v_prev[layer] = v_t
		return -v_t