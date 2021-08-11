import h5py
import os
import numpy as np

class Classifier:

    def __init__(self, N, M, name="classifier"):
        self.N = N
        self.M = M
        self.name = name

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def __call__(self, X):
        return self.predict(X)
    
    # returns the accuracy in the binary classification
    def accuracy(self, X, Y):
        return len(np.where(self.predict(X) == Y)[0])/len(Y)

    # the _save and _load methods will be an undisplayed method in each classifier
    def save(self, filename, absolute=False):
        path = os.path.join(os.getcwd(), filename) if not absolute else filename
        file = h5py.File(path+".h5", 'w')
        file.create_dataset('structure', (2,), np.ubyte, [self.N, self.M], compression="gzip")
        name_ASCII = np.array([ord(x) for x in self.name], np.ubyte) # name of the model saved as array of ASCII values
        file.create_dataset('name', name_ASCII.shape, np.ubyte, name_ASCII, compression="gzip")
        self._save(file)
        file.close()

    def load(self, filename, absolute=False):
        path = os.path.join(os.getcwd(), filename) if not absolute else filename
        file = h5py.File(path+".h5", 'r')
        self.N, self.M = file['structure']
        self.name = ''.join([chr(x) for x in file['name']])
        self._load(file)
        file.close()