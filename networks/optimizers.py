import numpy as np

# returns a basic instace (with the default parameters) of the optimizer given its type
def get_basic_optimizer_instance(optim_type):
	if optim_type == 1:
		return SGD()
	if optim_type == 2:
		return Momentum()
	if optim_type == 3:
		return Adagrad()
	if optim_type == 4:
		return Adadelta()
	return Adam()

class Optimizer:

	def __init__(self, name, lr, _type, decay, exp_decay):
		self.name = name
		self.lr = lr
		self._type = _type
		self.decay = decay
		self.exp_decay = exp_decay

	# the argument passed will be a one dimensional array with the number of parameters for each layer
	def compile(self, *args):
		pass
		
	# this method is only called from the instances of the class, which will all have their own
	# **kwargs, so this is just that, i don't want it to actually be the kwargs passed to this function
	def step(self, kwargs):
		epoch = 0 if 'epoch' not in kwargs else kwargs['epoch']
		return self.lr*np.exp(-self.decay*epoch) if self.exp_decay else self.lr/(1+self.decay*epoch)

	def __str__(self):
		return f"[Optimizer]\nName={self.name}, learning rate: {self.lr}\n"

	def __call__(self, grads, **kwargs):
		return self.step(grads)

	def save(self, file):
		group = file.create_group('optimizer')
		name = np.array([ord(x) for x in self.name])
		group.create_dataset('name', name.shape, np.ubyte, name, compression='gzip')
		group.create_dataset('lr', (3,), np.float32, [self.lr, self.decay, int(self.exp_decay)], compression='gzip')
		group.create_dataset('type', (1,), np.ubyte, [self._type], compression='gzip')

	def load(self, file):
		group = file['optimizer']
		self.name = ''.join([chr(x) for x in group['name']])
		self.lr, self.decay, self.exp_decay = group['lr']
		self.exp_decay = bool(self.exp_decay) # convert binary -> bool
		self._type = group['type'][0]

	@staticmethod
	def load_optimizer(file):
		_type = file['optimizer']['type'][0]
		optim = get_basic_optimizer_instance(_type)
		optim.load(file)
		return optim

	def __str__(self) -> str:
		decay_str = "exponential" if self.exp_decay else "kt"
		res = f'Optimizer: \n\tname: {self.name}\n\ttype: {type(self).__name__}\n\tlearning rate: {self.lr:.5f}'
		return res + (f"\n\tdecay: {self.decay:.5f}\n\tdecay type: {decay_str}" if self.decay != 0 else "")

class SGD(Optimizer):

	def __init__(self, lr=0.001, decay=0, exp_decay=True):
		super().__init__('Standard Gradient Descent', lr, 1, decay, exp_decay)

	def step(self, grads, **kwargs):
		lr = super().step(kwargs)
		return -lr*grads

class Momentum(Optimizer):

	def __init__(self, lr=0.001, momentum=0.1, decay=0, exp_decay=True):
		super().__init__('Momentum', lr, 2, decay, exp_decay)
		self.v_prev = None
		self.momentum = momentum

	def compile(self, params):
		self.v_prev = [np.zeros(layer_params) for layer_params in params]

	def step(self, grads, **kwargs):
		lr = super().step(kwargs)
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		v_t = self.momentum*self.v_prev[layer]+lr*grads
		self.v_prev[layer] = v_t
		return -v_t

	def save(self, file):
		super().save(file)
		group = file['optimizer']
		group.create_dataset('params', (1,), np.float32, [self.momentum], compression='gzip')
		for layer_idx, layer_mean in enumerate(self.v_prev):
			group.create_dataset(f'layer_{layer_idx}_mean', layer_mean.shape, np.float32, layer_mean, compression="gzip")

	def load(self, file):
		super().load(file)
		group = file['optimizer']
		self.momentum = group['params'][0]
		running_mean = []
		curr_layer = 0
		while f"layer_{curr_layer}_mean" in group.keys():
			running_mean.append(np.array(group[f"layer_{curr_layer}_mean"], dtype=np.float32))
			curr_layer += 1
		self.running_mean = running_mean

	def __str__(self) -> str:
		return super().__str__()+f"\n\tmomentum: {self.momentum:.5f}"

class Adagrad(Optimizer):

	def __init__(self, lr=0.2, decay=0, exp_decay=True):
		super().__init__('Adagrad', lr, 3, decay, exp_decay)
		self.running_mean = None

	def compile(self, num_params):
		self.running_mean = [np.zeros(layer_params) for layer_params in num_params]

	def step(self, grads, **kwargs):
		lr = super().step(kwargs)
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		self.running_mean[layer] += grads**2
		res = -lr*grads/np.sqrt(self.running_mean[layer]+1e-8)
		return res

	def save(self, file):
		super().save(file)
		group = file['optimizer']
		for layer_idx, layer_mean in enumerate(self.running_mean):
			group.create_dataset(f'layer_{layer_idx}_mean', layer_mean.shape, np.float32, layer_mean, compression="gzip")
		
	def load(self, file):
		super().load(file)
		group = file['optimizer']
		running_mean = []
		curr_layer = 0
		while f"layer_{curr_layer}_mean" in group.keys():
			running_mean.append(np.array(group[f"layer_{curr_layer}_mean"], dtype=np.float32))
			curr_layer += 1
		self.running_mean = running_mean

class Adadelta(Optimizer):

	def __init__(self, lr=0.001, gamma=0.99, initial_mean=1, decay=0, exp_decay=True):
		super().__init__('Adadelta', lr, 4, decay, exp_decay)
		self.gamma = gamma
		self.running_mean = None
		self.initial_mean = initial_mean

	def compile(self, num_params):
		self.running_mean = [np.zeros(layer_params) for layer_params in num_params]

	def step(self, grads, **kwargs):
		lr = super().step(kwargs)
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		self.running_mean[layer] = self.gamma*self.running_mean[layer] + (1-self.gamma)*(grads**2)
		res = -lr*grads/np.sqrt(self.running_mean[layer]+1e-8)
		return res

	def save(self, file):
		super().save(file)
		group = file['optimizer']
		group.create_dataset('params', (1,), np.float64, [self.gamma], compression="gzip")
		for layer_idx, layer_mean in enumerate(self.running_mean):
			group.create_dataset(f'layer_{layer_idx}_mean', layer_mean.shape, np.float32, layer_mean, compression="gzip")

	def load(self, file):
		super().load(file)
		group = file['optimizer']
		self.gamma = group['params'][0]
		running_mean = []
		curr_layer = 0
		while f"layer_{curr_layer}_mean" in group.keys():
			running_mean.append(np.array(group[f"layer_{curr_layer}_mean"], dtype=np.float32))
			curr_layer += 1
		self.running_mean = running_mean

	def __str__(self) -> str:
		return super().__str__() + f"\n\tgamma: {self.gamma:.5f}"

# TODO: implement RMSProp

class Adam(Optimizer):

	def __init__(self, lr=0.005, beta=0.9, gamma=0.999, decay=0, exp_decay=True):
		super().__init__('Adam', lr, 5, decay, exp_decay)
		self.beta = beta
		self.gamma = gamma

	def compile(self, num_params):
		self.m = [np.zeros(layer_params) for layer_params in num_params]
		self.v = [np.zeros(layer_params) for layer_params in num_params]

	def step(self, grads, **kwargs):
		lr = super().step(kwargs)
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		epoch = 1 if 'epoch' not in kwargs else kwargs['epoch']
		self.m[layer] = self.beta*self.m[layer]+(1-self.beta)*grads
		self.v[layer] = self.gamma*self.v[layer]+(1-self.gamma)*(grads**2)

		adj_lr = lr*np.sqrt(1-self.gamma**epoch)/(1-self.beta**epoch) # adjusted learning rate
		return -adj_lr*self.m[layer]/(np.sqrt(self.v[layer]+1e-8))

	def save(self, file):
		super().save(file)
		group = file['optimizer']
		group.create_dataset('params', (2,), np.float32, [self.beta, self.gamma], compression='gzip')
		for layer_idx in range(len(self.m)):
			group.create_dataset(f'layer_{layer_idx}_first_momentum', self.m[layer_idx].shape, np.float32, self.m[layer_idx], compression="gzip")
			group.create_dataset(f'layer_{layer_idx}_second_momentum', self.v[layer_idx].shape, np.float32, self.v[layer_idx], compression="gzip")

	def load(self, file):
		super().load(file)
		group = file['optimizer']
		self.beta, self.gamma = group['params']
		m, v = [], []
		curr_layer = 0
		while f'layer_{curr_layer}_first_momentum' in group.keys():
			m.append(np.array(group[f'layer_{curr_layer}_first_momentum'], dtype=np.float32))
			v.append(np.array(group[f'layer_{curr_layer}_second_momentum'], dtype=np.float32))
			curr_layer += 1
		self.m, self.v = m, v

	def __str__(self) -> str:
		return super().__str__() + f"\n\tbeta: {self.beta:.5f}\n\tgamma: {self.gamma:.5f}"