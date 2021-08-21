import numpy as np
from classification.classifier import Classifier

class NaiveBayes(Classifier):

    def __init__(self, name='Naive Bayes'):
        super().__init__(1,1,name,_type=3)
        self.name = name

    def _predict(self, x):
        if len(self._labels) == 0:
            print("It is necessary to train the classifier before using it to make predictions")
            return
        # it might look complex but it's just prior + posterior probabilities for each label
        probs = np.array([np.log(self.P_C[i])+np.sum(np.log(self.gaussian(x,i))) for i in range(self.M)])
        return self._labels[np.argmax(probs)]

    def fit(self, X, Y, *args, **kwargs):
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        self._labels = np.unique(Y)
        self.N = X.shape[1]
        self.M = len(self._labels)
        self.mean = np.zeros(X.shape)
        self.var = np.zeros(X.shape)
        self.P_C = np.array([len(np.where(Y==i)[0])/float(len(Y)) for i in self._labels])

        # for each class
        for idx, cl in enumerate(self._labels):
            indexes = np.where(Y==cl)[0]
            self.mean[idx,: ] = np.mean(X[indexes], axis=0)
            self.var[idx,:] = np.var(X[indexes], axis=0)

    def gaussian(self, x, idx):
        return np.exp(-((x-self.mean[idx])**2/(2*self.var[idx])))/np.sqrt(2*np.pi*self.var[idx])

    def _save(self, file):
        file.create_dataset('labels', self._labels.shape, self._labels.dtype, self._labels, compression="gzip")
        file.create_dataset('mean', self.mean.shape, self.mean.dtype, self.mean, compression="gzip")
        file.create_dataset('variance', self.var.shape, self.var.dtype, self.var, compression="gzip")
        file.create_dataset('priors', self.P_C.shape, self.P_C.dtype, self.P_C, compression="gzip")

    def _load(self, file):
        self._labels = np.array(file['labels'])
        self.mean = np.array(file['mean'])
        self.var = np.array(file['variance'])
        self.P_C = np.array(file['priors'])