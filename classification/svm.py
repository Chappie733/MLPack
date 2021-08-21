import numpy as np
from cvxopt import matrix, solvers
from classification.classifier import Classifier

# Lagrange multipliers threshold
DEFAULT_LM_THRESHOLD = 1e-4
KERNELS = {}

def logger(kernel):
    global KERNELS
    KERNELS[kernel.__name__] = kernel
    return kernel

@logger
def linear(x,y, **kwargs):
    return np.dot(x,y)

@logger
def polynomial(x,y, **kwargs):
    return (np.dot(x,y)+kwargs['c'])**kwargs['n']

@logger
def gaussian(x,y, **kwargs):
    return np.exp(-np.linalg.norm(x-y)/(2*kwargs['stddev']**2))

@logger
def rbf(x,y, **kwargs):
    return np.exp(-kwargs['gamma']*np.linalg.norm(x-y))

@logger
def sigmoid(x,y, **kwargs):
    return np.tanh(kwargs['gamma']*np.dot(x,y)+kwargs['c'])

class SVM(Classifier):

    def __init__(self, kernel='linear', name='SVM', **kwargs):
        super().__init__(1, 2, name, _type=2)
        self.kernel = kernel if not isinstance(kernel, str) else KERNELS[kernel]
        self.name = name
        self.c = 0 if 'c' not in kwargs else kwargs['c']
        self.stddev = 1 if 'stddev' not in kwargs else kwargs['stddev']
        self.n = 1 if 'n' not in kwargs else kwargs['n']
        self.gamma = 1 if 'gamma' not in kwargs else kwargs['gamma']
        self.threshold = DEFAULT_LM_THRESHOLD if 'threshold' not in kwargs else kwargs['threshold']

    def _predict(self, x):
        s = self.bias
        for i in range(len(self.alphas)):
            s += self.alphas[i]*self.Y[i]*self.kernel(self.X[i], x, c=self.c, stddev=self.stddev, n=self.n, gamma=self.gamma)
        return 1 if s >= 0 else -1 # s could be 0, np.sign(0) = 0

    def fit(self, X, Y, verbose=True, *args, **kwargs):
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        p = len(X)
        self.X = X
        self.Y = Y

        self.N = X.shape[1]

        P = matrix(0.0, (p,p))
        for j in range(p):
            for i in range(p):
                P[i,j] = self.kernel(X[i], X[j], c=self.c, stddev=self.stddev, n=self.n, gamma=self.gamma)
        Y_mat = np.vstack([Y]*p)

        P = matrix(P*(Y_mat*Y_mat.T))
        q = matrix(-np.ones(shape=(p,)))
        h = matrix(np.zeros(p,))
        G = matrix(-np.eye(p))
        b = matrix(0.0, (1,1))
        A = matrix(np.reshape(Y, (1,p)))

        solvers.options['show_progress'] = verbose
        self.alphas = np.ravel(solvers.qp(P, q, G, h, A, b)['x'])

        indices = np.where(self.alphas >= self.threshold)
        self.X = X[indices]
        self.Y = Y[indices]
        self.alphas = self.alphas[indices]
        j = indices[0][0]

        # calculate the bias
        sum_term = 0
        for i in range(len(self.alphas)):
            sum_term += self.alphas[i]*self.Y[i]*self.kernel(self.X[i], X[j], c=self.c, stddev=self.stddev, n=self.n, gamma=self.gamma)

        self.bias = Y[j]-sum_term
        
    # Hinge loss
    def loss(self, Y, predictions):
        return np.sum(np.maximum(1-Y*predictions, 0))

    def _save(self, file):
        file.create_dataset('alphas', self.alphas.shape, np.float32, self.alphas, compression="gzip")
        file.create_dataset('bias', (1,), np.float32, self.bias, compression="gzip")
        file.create_dataset('data_X', self.X.shape, np.float32, self.X, compression="gzip")
        file.create_dataset('data_Y', self.Y.shape, np.float32, self.Y, compression="gzip")
        file.create_dataset('params', (5,), np.float32, [self.c, self.stddev, self.n, self.gamma, self.threshold], compression="gzip")
        kernel_name_ASCII = np.array([ord(x) for x in self.kernel.__name__], dtype=np.ubyte)
        file.create_dataset('kernel', kernel_name_ASCII.shape, np.ubyte, kernel_name_ASCII, compression="gzip")

    def _load(self, file):
        self.alphas = np.array(file['alphas'])
        self.bias = file['bias'][0]
        self.X = np.array(file['data_X'], dtype=np.float32)
        self.Y = np.array(file['data_Y'], dtype=np.float32)
        self.c, self.stddev, self.n, self.gamma, self.threshold = file['params']
        self.name = ''.join([chr(x) for x in file['name']])
        self.kernel = KERNELS[''.join([chr(x) for x in file['kernel']])]