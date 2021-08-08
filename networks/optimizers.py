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

	def __init__(self, name, lr, _type):
		self.name = name
		self.lr = lr
		self._type = _type

	def compile(self, *args):
		pass

	def __str__(self):
		return f"[Optimizer]\nName={self.name}, learning rate: {self.lr}\n"

	def save(self, file):
		group = file.create_group('optimizer')
		name = np.array([ord(x) for x in self.name])
		group.create_dataset('name', name.shape, np.ubyte, name, compression='gzip')
		group.create_dataset('lr', (1,), np.float32, [self.lr], compression='gzip')
		group.create_dataset('type', (1,), np.ubyte, [self._type], compression='gzip')

	def load(self, file):
		group = file['optimizer']
		self.name = ''.join([chr(x) for x in group['name']])
		self.lr = group['lr'][0]
		self._type = group['type'][0]

class SGD(Optimizer):

	def __init__(self, lr=0.001):
		super().__init__('Standard Gradient Descent', lr, 1)

	def step(self, grads, **kwargs):
		return -self.lr*grads

	def __call__(self, grads, **kwargs):
		return self.step(grads)

class Momentum(Optimizer):

	def __init__(self, lr=0.001, momentum=0.1):
		super().__init__('Momentum', lr, 2)
		self.v_prev = None
		self.momentum = momentum

	def compile(self, struct):
		self.v_prev = [np.zeros(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		v_t = self.momentum*self.v_prev[layer]+self.lr*grads
		self.v_prev[layer] = v_t
		return -v_t

	def __str__(self):
		return f"[Optimizer]\nName={self.name}, learning rate: {self.lr}, momentum: {self.momentum}\n"

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

class Adagrad(Optimizer):

	def __init__(self, lr=0.2, initial_mean=1.0):
		super().__init__('Adagrad', lr, 3)
		self.running_mean = None
		self.initial_mean = 1.0

	def compile(self, struct):
		self.running_mean = [np.ones(shape=(struct[l+1]*(struct[l]+1)))*self.initial_mean for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		self.running_mean[layer] += grads**2
		res = -self.lr*grads/np.sqrt(self.running_mean[layer]+1e-8)
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

	def __init__(self, lr=0.001, gamma=0.99, initial_mean=1):
		super().__init__('Adadelta', lr, 4)
		self.gamma = gamma
		self.running_mean = None
		self.initial_mean = initial_mean

	def compile(self, struct):
		self.running_mean = [np.ones(shape=(struct[l+1]*(struct[l]+1)))*self.initial_mean for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		self.running_mean[layer] = self.gamma*self.running_mean[layer] + (1-self.gamma)*(grads**2)
		res = -self.lr*grads/np.sqrt(self.running_mean[layer]+1e-8)
		return res

	def __str__(self):
		return f"[Optimizer]\nName={self.name}, learning rate: {self.lr}, gamma={self.gamma}\n"

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

class Adam(Optimizer):

	def __init__(self, lr=0.005, beta=0.9, gamma=0.999):
		super().__init__('Adam', lr, 5)
		self.beta = beta
		self.gamma = gamma

	def compile(self, struct):
		self.m = [np.zeros(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]
		self.v = [np.zeros(shape=(struct[l+1]*(struct[l]+1))) for l in range(len(struct)-1)]

	def step(self, grads, **kwargs):
		layer = 0 if 'layer' not in kwargs else kwargs['layer']
		epoch = 1 if 'epoch' not in kwargs else kwargs['epoch']
		self.m[layer] = self.beta*self.m[layer]+(1-self.beta)*grads
		self.v[layer] = self.gamma*self.v[layer]+(1-self.gamma)*(grads**2)

		v_norm = self.v[layer]/(1-self.gamma**epoch)
		m_norm = self.m[layer]/(1-self.beta**epoch)

		return -self.lr*m_norm/(np.sqrt(v_norm)+1e-8)

	def __str__(self):
		return f"[Optimizer]\nName={self.name}, learning rate: {self.lr}, beta_1={self.beta}, beta_2={self.gamma}\n"

	def save(self, file):
		super().save(file)
		group = file['optimizer']
		group.create_dataset('params', (2,). np.float32, [self.beta, self.gamma], compression='gzip')
		for layer_idx in range(len(self.m)):
			group.create_dataset(f'layer_{layer_idx}_first_momentum', self.m[layer_idx].shape, np.float32, self.m[layer_idx], compression="gzip")
			group.create_dataset(f'layer_{layer_idx}_second_momentum', self.v[layer_idx], np.float32, self.v[layer_idx], compression="gzip")

	def load(self, file):
		super().load(file)
		group = file['optimizer']
		self.beta, self.gamma = group['params']
		m, v = [], []
		curr_layer = 0
		while f'layer_{curr_layer}_first_momentum' in group.keys():
			m.append(np.array(file[f'layer_{curr_layer}_first_momentum'], dtype=np.float32))
			v.append(np.array(file[f'layer_{curr_layer}_second_momentum'], dtype=np.float32))
			curr_layer += 1
		self.m, self.v = m, v