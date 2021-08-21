import h5py
import os
import numpy as np
from classification import *
import classification

def get_basic_classifier_instance(clf_type):
    if clf_type == 1:
        return classification.Perceptron(1)
    elif clf_type == 2:
        return classification.SVM()
    elif clf_type == 3:
        return classification.NaiveBayes()
    elif clf_type == 4:
        return classification.KNN()
    elif clf_type == 5:
        return classification.LinearMachine(1,1)

class Classifier:

    def __init__(self, N, M, name="classifier", _type=0):
        self.N = N
        self.M = M
        self.name = name
        self._type = _type

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def __call__(self, X):
        return self.predict(X)
    
    # returns the accuracy in the binary classification
    def accuracy(self, X, Y) -> float:
        return len(np.where(self.predict(X) == Y)[0])/len(Y)

    # the _save and _load methods will be an undisplayed method in each classifier
    def save(self, filename: str, absolute=False) -> None:
        path = os.path.join(os.getcwd(), filename) if not absolute else filename
        file = h5py.File(path+".h5", 'w')
        file.create_dataset('structure', (3,), np.ubyte, [self.N, self.M, self._type], compression="gzip")
        name_ASCII = np.array([ord(x) for x in self.name], np.ubyte) # name of the model saved as array of ASCII values
        file.create_dataset('name', name_ASCII.shape, np.ubyte, name_ASCII, compression="gzip")
        self._save(file)
        file.close()

    def save_in_file(self, file: h5py.File) -> None:
        file.create_dataset('structure', (3,), np.ubyte, [self.N, self.M, self._type], compression="gzip")
        name_ASCII = np.array([ord(x) for x in self.name], np.ubyte) # name of the model saved as array of ASCII values
        file.create_dataset('name', name_ASCII.shape, np.ubyte, name_ASCII, compression="gzip")
        self._save(file)

    def load(self, filename: str, absolute=False) -> None:
        path = os.path.join(os.getcwd(), filename) if not absolute else filename
        file = h5py.File(path+".h5", 'r')
        self.N, self.M, self._type = file['structure']
        self.name = ''.join([chr(x) for x in file['name']])
        self._load(file)
        file.close()

    def load_from_file(self, file: h5py.File) -> None:
        self.N, self.M, self._type = file['structure']
        self.name = ''.join([chr(x) for x in file['name']])
        self._load(file)

    # this can be called to load a classifier from a file without already having an instance of it
    @staticmethod
    def load_classifier(filename: str, absolute=False):
        path = os.path.join(os.getcwd(), filename) if not absolute else filename
        file = h5py.File(path+".h5", 'r')
        clf_type = file['structure'][2]
        classifier = get_basic_classifier_instance(clf_type)
        classifier.load_from_file(file)
        file.close()
        return classifier
    
    @staticmethod
    def load_classifier(file: h5py.File):
        clf_type = file['structure'][2]
        classifier = get_basic_classifier_instance(clf_type)
        classifier.load_from_file(file)
        return classifier