import numpy as np
from cvxopt import matrix, solvers

# Lagrange multipliers threshold
DEFAULT_LM_THRESHOLD = 1e-4

def linear(x,y, **kwargs):
    return np.dot(x,y)

def polynomial(x,y,**kwargs):
    return (np.dot(x,y)+kwargs['c'])**kwargs['n']

def gaussian(x,y, **kwargs):
    return np.exp(-np.linalg.norm(x-y)/(2*kwargs['stddev']**2))

def rbf(x,y, **kwargs):
    return np.exp(-kwargs['gamma']*np.linalg.norm(x-y))

def sigmoid(x,y, **kwargs):
    return np.tanh(gamma*np.dot(x,y)+kwargs['c'])

class SVM:

    def __init__(self, kernel=linear, **kwargs):
        self.kernel = kernel

        self.c = 0 if 'c' not in kwargs else kwargs['c']
        self.stddev = 1 if 'stddev' not in kwargs else kwargs['stddev']
        self.n = 1 if 'n' not in kwargs else kwargs['n']
        self.gamma = 1 if 'gamma' not in kwargs else kwargs['gamma']
        self.threshold = DEFAULT_LM_THRESHOLD if 'threshold' not in kwargs else kwargs['threshold']

    def _predict(self, x):
        s = self.bias
        for i in range(len(self.alphas)):
            s += self.alphas[i]*self.Y[i]*self.kernel(self.X[i], x, c=self.c, stddev=self.stddev, n=self.n, gamma=self.gamma)
        return 1 if s > 0 else -1 # s could be 0, np.sign(0) = 0

    def predict(self, X):
        return [self._predict(x) for x in X]

    def fit(self, X, Y, verbose=True):
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        p = len(X)
        self.X = X
        self.Y = Y

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


SAMPLE_SIZE = 4
FEATURES = 3
epochs = 1000

X = np.random.uniform(low=-1,high=1, size=(SAMPLE_SIZE, FEATURES))
Y = np.sign(np.random.uniform(low=-1, high=1, size=(SAMPLE_SIZE,)))
Y[Y==0] = 1

model = SVM(kernel=polynomial)
model.fit(X,Y)

preds = model.predict(X[:10])
outs = Y[:10]

pred_classes = ["A" if x == 1 else "B" for x in preds]
outs_classes = ["A" if x == 1 else "B" for x in outs]

print("Predictions: " + ', '.join(pred_classes))
print("Correct classification: " + ', '.join(outs_classes))

print("\nAmount of Support vectors: %i" % len(np.where(model.alphas!=0)))