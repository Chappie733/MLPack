import numpy as np
import h5py
import os

from .layers import Layer
from networks.errors import ErrorFunction
from networks.optimizers import Optimizer

class Model():
    '''
        A class representing a standard sequential neural network.\n
        layers -> A list of "Layer" objects specifying in the right order the layers of the model.
    '''

    def __init__(self, layers, name="neural network") -> None:
        self.layers = layers
        self.num_layers = len(layers)
        self.name = name

    def compile(self, error_func: ErrorFunction, optimizer: Optimizer) -> None:
        '''
            Compiles (Initializes) the model with the given error function (error_func)
            and optimizer.\n
            error_func -> The error function the network minimizes.\n
            optimizer -> The optimizer used in the gradient descent process.
        '''
        for layer in self.layers:
            layer.compile()
        self.error_func = error_func
        self.optimizer = optimizer
        self.optimizer.compile([layer.get_num_params() for layer in self.layers])

    def _predict(self, x, save_vals=True) -> np.ndarray:
        '''
            Returns the prediction of the network on the input vector x.\n
            save_vals -> whether the values of the neurons of each layer are saved or not.
        '''
        for layer in self.layers:
            x = layer.forward(x, save_vals=save_vals)
        return x

    def predict(self, X) -> np.ndarray:
        '''
            Returns an array where at the i-th index is the prediction of the
            network for the input vector X[i].
        '''
        return np.array([self._predict(x, save_vals=False) for x in X])

    def fit_stochastic(self, X, Y, epochs=5, verbose=True, return_errors=False):
        '''
            Trains the network on the inputs X and outputs Y for the given number of epochs.\n
            verbose -> Whether or not to print the loss after each epoch.\n
            return_errors -> Whether or not to return a list with the values of the loss function
            at each epoch.
        '''
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y, dtype=np.float32)

        if return_errors:
            errs = []

        for epoch in range(epochs):
            for u in range(len(Y)):
                pred = self._predict(X[u])

                errors = self.error_func.grad(Y[u], pred)
                if self.layers[-1].activation.__name__ == 'softmax':
                    out_matrix = np.vstack([pred]*len(pred))
                    Z = out_matrix*out_matrix.T-out_matrix*np.eye(len(pred))
                    errors = -np.dot(Z, errors)

                for l_idx in range(self.num_layers-1, -1, -1):
                    gradients = self.layers[l_idx].get_gradients(errors)
                    updates = self.optimizer.step(gradients, layer=l_idx)
                    errors = self.layers[l_idx].backward(errors) 
                    self.layers[l_idx].update_params(updates)
    
            if verbose or return_errors:
                predictions = self.predict(X)
                err = self.error_func(Y, predictions)
                if verbose:
                    print(f"Epoch #{epoch+1}, loss: {err}")
                if return_errors:
                    errs.append(err)
        
        return None if not return_errors else errs

    def fit(self, X, Y, epochs=5, batch_size=32, verbosity=1, return_errors=0):
        '''
            Trains the network on the inputs X and outputs Y for the given number of epochs and
            with the given batch size.\n
            verbosity -> How much of the training process is printed on the console\n
            \tverbosity = 0 -> Nothing is printed\n
            \tverbosity = 1 -> The loss is printed at each epoch\n
            \tverbosity = 2 -> The loss is printed at each epoch and at each batch\n
            returns_errors -> How frequently the errors are saved and returned during the training process.\n
            \treturn_errors = 0 -> The errors are not logged or returned.\n
            \treturn_errors = 1 -> The errors are logged for every epoch.\n
            \treturn_errors = 2 -> The errors are logged for every batch.\n
            All of the errors are subsequently returned by the function in a list
        '''
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y, dtype=np.float32)

        n_batches = int(np.ceil(len(Y)/batch_size))

        epoch_loss_print, batch_loss_print = verbosity >= 1, verbosity >= 2

        if return_errors != 0:
            errs = []

        for epoch in range(epochs):
            for batch in range(n_batches):
                batch_start = batch*batch_size
                batch_end = min((batch+1)*batch_size, len(Y))

                batch_updates = [np.zeros(layer.get_num_params()) for layer in self.layers]

                if batch_loss_print or return_errors == 2:
                    avg_batch_loss = 0

                for u in range(batch_start, batch_end):
                    pred = self._predict(X[u])

                    if batch_loss_print:
                        avg_batch_loss += self.error_func(Y[u], pred)

                    errors = self.error_func.grad(Y[u], pred)
                    if self.layers[-1].activation.__name__ == 'softmax':
                        out_matrix = np.vstack([pred]*len(pred))
                        Z = out_matrix*out_matrix.T-out_matrix*np.eye(len(pred))
                        errors = -np.dot(Z, errors)

                    for l_idx in range(self.num_layers-1, -1, -1):
                        gradients = self.layers[l_idx].get_gradients(errors)
                        updates = self.optimizer.step(gradients, layer=l_idx)
                        errors = self.layers[l_idx].backward(errors)
                        batch_updates[l_idx] += updates
                
                for l_idx in range(self.num_layers):
                    self.layers[l_idx].update_params(batch_updates[l_idx]/batch_size)

                if batch_loss_print:
                    print(f"\tBatch {batch+1}/{n_batches}, loss = {avg_batch_loss/batch_size}")
                if return_errors == 2:
                    errs.append(avg_batch_loss/batch_size)

            if epoch_loss_print or return_errors == 1:
                predictions = self.predict(X)
                err = self.error_func(Y, predictions)
                if return_errors == 1:
                    errs.append(err)
                if epoch_loss_print:
                    print(f"Epoch {epoch+1}/{epochs}, loss: {err}")

        return None if return_errors == 0 else errs

    def save(self, filename: str, copts=4, absolute=False, save_state=False) -> None:
        '''
            Saves the model and all of its parameters in the file with name "filename.h5" in the current
            working directory.\n
            filename -> The name of the file the model will be saved in.\n
            copts -> The amount of compression applied from 0 (lowest) to 9 (highest).\n
            absolute -> Whether the path passed in filename is absolute or relative (under the current 
            working directory).\n
            save_state -> Whether the values of the neurons in the layers of the network are also saved or not.
        '''
        path = os.path.join(os.getcwd(), filename) if not absolute else filename
        file = h5py.File(path+'.h5', 'w')
        for i in range(self.num_layers):
            self.layers[i].save(file, i, save_state=save_state, copts=4)
        name_ASCII = [ord(x) for x in self.name]
        file.create_dataset('name', (len(self.name)), np.ubyte, name_ASCII, compression="gzip", compression_opts=copts)
        self.optimizer.save(file, copts)
        file.close()

    def load(self, filename: str, absolute=False) -> None:
        '''
            Loads the model from the file with name "filename.h5" under the current working directory.\n
            filename -> The name of the file in which the model is saved.\n
            absolute -> Whether filename is an absolute path to the file or not.
        '''
        path = os.path.join(os.getcwd(), filename) if not absolute else filename
        file = h5py.File(path+'.h5', 'r')
        self.name = ''.join([chr(x) for x in file['name']])
        layers = []
        curr_layer = 0
        while f'layer_{curr_layer}' in file.keys():			
            layers.append(Layer.load_layer(file, curr_layer))
            curr_layer += 1
        self.num_layers = len(layers)
        self.layers = layers
        self.optimizer = Optimizer.load_optimizer(file)
        file.close()

    def __str__(self):
        res = f"Sequential Model: name: {self.name}, number of layers: {self.num_layers}\n"
        res += "-"*40+"\n"
        trainable_params = 0
        for i in range(self.num_layers):
            res += str(self.layers[i])
            trainable_params += self.layers[i].get_num_trainable_params()
        res += f"Trainable params: {trainable_params}\n"
        res += str(self.optimizer)
        return res

    def __getitem__(self, index) -> Layer:
        return self.layers[index]