import numpy as np

class Optimizer:

	def __init__(self, name, lr):
		self.name = name
		self.lr = lr

	def compile(self, *args):
		pass

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

	def compile(self, struct):
		self.v_prev = [np.zeros(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		v_t = self.momentum*self.v_prev[layer]+self.lr*grads
		self.v_prev[layer] = v_t
		return -v_t

class Adagrad(Optimizer):

	def __init__(self, lr=0.2):
		super().__init__('Adagrad', lr)
		self.running_mean = None

	def compile(self, struct):
		self.running_mean = [np.ones(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		self.running_mean[layer] += grads**2
		res = -self.lr*grads/np.sqrt(self.running_mean[layer]+1e-8)
		return res

class Adadelta(Optimizer):

	def __init__(self, lr=0.001, gamma=0.99):
		super().__init__('Adadelta', lr)
		self.gamma = gamma
		self.running_mean = None

	def compile(self, struct):
		self.running_mean = [np.ones(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		self.running_mean[layer] = self.gamma*self.running_mean[layer] + (1-self.gamma)*(grads**2)
		res = -self.lr*grads/np.sqrt(self.running_mean[layer]+1e-8)
		return res

class Adam(Optimizer):

	def __init__(self, lr=0.005, beta=0.9, gamma=0.999):
		super().__init__('Adam', lr)
		self.beta = beta
		self.gamma = gamma

	def compile(self, struct):
		self.m = [np.ones(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]
		self.v = [np.ones(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		epoch = 1 if 'epoch' not in kwargs else kwargs['epoch']
		self.m[layer] = self.beta*self.m[layer]+(1-self.beta)*grads
		self.v[layer] = self.gamma*self.v[layer]+(1-self.gamma)*(grads**2)

		v_norm = self.v[layer]/(1-self.gamma**epoch)
		m_norm = self.m[layer]/(1-self.beta**epoch)

		return -self.lr*m_norm/(np.sqrt(v_norm)+1e-8)
