import numpy as np

class NaiveBayes:

    def __init__(self, name=None):
        self.name = name

    def _predict(self, x):
        if len(self._labels) == 0:
            print("It is necessary to train the classifier before using it to make predictions")
            return
        # it might look complex but it's just prior + posterior probabilities for each label
        probs = np.array([np.log(self.P_C[i])+np.sum(np.log(self.gaussian(x,i))) for i in range(self.n_labels)])
        return self._labels[np.argmax(probs)]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def fit(self, X, Y):
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        self._labels = np.unique(Y)
        self.n_labels = len(self._labels)
        n_samples, n_features = X.shape
        self.mean = np.zeros((n_samples, n_features))
        self.var = np.zeros((n_samples, n_features))
        self.P_C = np.array([len(np.where(Y==i)[0])/float(len(Y)) for i in self._labels])

        # for each class
        for idx, cl in enumerate(self._labels):
            indexes = np.where(Y==cl)[0]
            self.mean[idx,: ] = np.mean(X[indexes], axis=0)
            self.var[idx,:] = np.var(X[indexes], axis=0)

    def gaussian(self, x, idx):
        return np.exp(-((x-self.mean[idx])**2/(2*self.var[idx])))/np.sqrt(2*np.pi*self.var[idx])