from typing import ForwardRef
import numpy as np
from numpy.core.getlimits import _register_known_types
from networks.activations import ACTIVATIONS
from h5py import File

class Layer():
    '''
        A class representing a general layer of a network, this is a dense layer by default, but
        many other classes inherit from this one.
    '''
    
    def __init__(self, num_neurons: int, num_outs: int, activation="linear", name="Dense", k=1, _type=1) -> None:
        '''
            num_neurons -> the number of neurons in the layer\n
            num_outs -> the number of values the layer outputs\n
            activation -> the activation function the layer applies after the linear transformation\n
            name -> the name of the layer, this is written when the layer is printed\n
            k -> constant deciding the variance with which the weights are intialized, which is σ² = k/num_neurons
        '''
        self.num_neurons = num_neurons
        self.num_outs = num_outs
        self.name = name
        self.k = k
        self._type = _type
        if isinstance(activation, str):
            global ACTIVATIONS
            self.activation = ACTIVATIONS[activation.lower()]
        else:
            self.activation = activation
    
    def compile(self) -> None:
        '''
            Initializes the parameters of the network, as well as all the data about its nodes
        '''
        stddev = np.sqrt(self.k/float(self.num_neurons))
        self.weights = np.random.normal(scale=stddev, size=(self.num_outs, self.num_neurons)) # mean is 0 by default
        self.neurons = np.zeros(self.num_neurons)
        self.bias = np.zeros(self.num_outs)

    def get_local_fields(self, *args) -> np.ndarray:
        '''
            Returns the local fields this layer generates (feeding into the next layer)
        '''
        return np.dot(self.weights, self.neurons)+self.bias

    def forward(self, x, save_vals=True, **kwargs) -> np.ndarray:
        '''
            Forward pass of the layer
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError("Expected numpy array but received {type} instead".format(type=type(x)))
        if save_vals:
            self.neurons = x
        return self.activation(np.dot(self.weights, x)+self.bias)

    def backward(self, errors: np.ndarray, *args) -> np.ndarray:
        '''
            Backward pass of the layer, given the errors of the next layer this returns the errors
            to be passed to the previous layer
        '''
        return np.dot(self.weights.T, errors*self.activation(self.get_local_fields(), deriv=True))

    def get_gradients(self, errors: np.ndarray, *args) -> np.ndarray:
        '''
            Given the gradients of the errors of the following layer this function returns a numpy
            array containing all the gradients of the error function with respect to the parameters
            in this layer
        '''
        grads = np.zeros((self.num_outs, self.num_neurons))
        errors = errors*self.activation(self.get_local_fields(), deriv=True)
        for n in range(self.num_outs):
            for m in range(self.num_neurons):
                grads[n][m] = self.neurons[m]*errors[n]

        # this automatically reshapes the array so it's one dimensional, the first num_out*num_neurons
        # entries are the gradients of the weights
        grads = np.append(grads, errors)
        return grads

    def update_params(self, updates: np.ndarray) -> None:
        '''
            Updates the parameters according to the updates vector, which should be the result of
            Layer.get_gradients() after being processed by an optimizer
        '''
        self.weights = self.weights + updates[:-self.num_outs].reshape((self.num_outs, self.num_neurons))
        self.bias = self.bias + updates[-self.num_outs:]

    def get_out_deriv(self) -> np.ndarray:
        '''
            Returns the derivative of the local fields this layer computes
        '''
        return self.activation(self.get_local_fields(), deriv=True)

    def get_num_params(self) -> int:
        '''
            Returns the number of parameters in the layer
        '''
        return self.num_outs*(self.num_neurons+1)

    def get_num_trainable_params(self) -> int:
        '''
            Returns the number of trainable parameters in the layer
        '''
        return self.get_num_params()

    def save(self, file: File, layer_idx: int, save_state=False, copts=4) -> None:
        '''
            Saves the layer on the h5py.File instance file, under a new group called "layer_l" where l
            is the value passed for l_idx.\n
            save_state -> whether the values of the neurons of the layer are saved.\n
            copt -> the level of compression adopted on the values (0-9).
        '''
        group = file.create_group(f'layer_{layer_idx}')
        group.create_dataset('weights', self.weights.shape, np.float32, self.weights, compression="gzip", compression_opts=copts)
        group.create_dataset('bias', self.bias.shape, np.float32, self.bias, compression="gzip", compression_opts=copts)
        group.create_dataset('params', (2,), np.ubyte, [self.k, self._type], compression="gzip", compression_opts=copts)
        # an array containing the name of the layer (at each index is the characters ASCII CODE)
        name_ASCII = [ord(x) for x in self.name]
        group.create_dataset('name', (len(self.name)), np.ubyte, name_ASCII, compression="gzip", compression_opts=copts)

        activation_ASCII = [ord(x) for x in self.activation.__name__]
        group.create_dataset('activation', (len(self.activation.__name__)), np.ubyte, activation_ASCII, compression="gzip", compression_opts=copts)
        if save_state:
            group.create_dataset('neurons', self.weights.shape, np.float32, self.weights, compression="gzip", compression_opts=copts)

    def load(self, file: File, layer_idx: int) -> None:
        '''
            Loads the layer from the group called "layer_l" (where l is layer_idx) in the h5py.File instance file.
        '''
        if not self.is_compiled():
            self.compile()
        global ACTIVATIONS

        group = file[f'layer_{layer_idx}']
        self.k, self._type = group['params']
        self.weights = np.array(group['weights'], dtype=np.float32)
        self.bias = np.array(group['bias'], dtype=np.float32)
        self.name = ''.join([chr(x) for x in group['name']])
        self.num_outs, self.num_neurons = self.weights.shape
        
        activation_name = ''.join([chr(x) for x in group['activation']]).lower()
        self.activation = ACTIVATIONS[activation_name]
        self.neurons = np.zeros(self.num_neurons, dtype=np.float32)
        if 'neurons' in group.keys():
            self.neurons = np.array(group['neurons'], dtype=np.float32)

    @staticmethod
    def load_layer(file: File, layer_idx: int) -> None:
        '''
            Loads the layer saved in the group 'layer_l' (where l is layer_idx) in the given h5py.File 
            instance file.
        '''
        global ACTIVATIONS
        group = file[f'layer_{layer_idx}']
        k, _type = group['params']
        if _type == 1:
            layer = Layer(1,1)
            layer.load(file, layer_idx)
            return layer

    def is_compiled(self) -> bool:
        '''
            Returns whether the layer has been compiled or not
        '''
        return 'weights' in dir(self) and 'neurons' in dir(self) and 'bias' in dir(self)

    def __call__(self, x, save_vals=True):
        return self.forward(x, save_vals=save_vals)

    def __str__(self) -> str:
        return f"{self.name} [neurons:{self.num_neurons}, k={self.k}, activation={self.activation.__name__}]\n"+"-"*40+"\n"


class Dropout(Layer):

    def __init__(self, num_neurons: int, num_outs: int, rate=0.1, activation="linear", name="Dropout", k=1) -> None:
        super().__init__(num_neurons, num_outs, activation=activation, name=name, k=k, _type=2)
        self.rate = rate
    
    def compile(self):
        super().compile()
        # this matrix has the same size of the weights one and it determines 
        # which weights are active (1) and which ones aren't (0)
        self.fixed = np.random.choice(a=[0, 1], size=(self.num_outs, self.num_neurons), p=[self.rate, 1-self.rate])
        self.weights *= self.fixed

    def get_gradients(self, errors: np.ndarray) -> np.ndarray:
        grads = np.zeros((self.num_outs, self.num_neurons))
        errors = errors*self.activation(self.get_local_fields(), deriv=True)
        for n in range(self.num_outs):
            for m in range(self.num_neurons):
                grads[n][m] = self.neurons[m]*errors[n]

        grads = np.append(grads*self.fixed, errors)
        return grads

    def update_params(self, updates: np.ndarray) -> None:
        self.weights = self.weights + updates[:-self.num_outs].reshape((self.num_outs, self.num_neurons))*self.fixed
        self.bias = self.bias + updates[-self.num_outs:]

    def get_num_trainable_params(self) -> int:
        return (self.num_outs+1)*self.num_neurons-len(self.fixed[self.fixed==0])

    def save(self, file: File, layer_idx: int, save_state=False, copts=4) -> None:
        super().save(file, layer_idx, save_state=save_state, copts=copts)
        group = file[f'layer_{layer_idx}']
        group.create_dataset('fixed', self.fixed.shape, np.float32, self.fixed, compression="gzip", compression_opts=copts)
        group.create_dataset('rate', (), np.float32, self.rate, compression="gzip", compression_opts=copts)

    def load(self, file: File, layer_idx: int) -> None:
        super().load(file, layer_idx)
        group = file[f'layer_{layer_idx}']
        self.fixed = np.array(group['fixed'], dtype=np.float32)
        self.rate = group['rate']