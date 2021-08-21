import numpy as np
import h5py
from enum import Enum
from copy import deepcopy
from classification.classifier import Classifier

class Approach(Enum):
    ONE_VS_ALL = 1
    CLASS_IS_REACHED = 2

'''
A boosting algorithm that uses a reference binary classifier
to tackle non binary classification problems 
'''
class BinaryClassifierAugmenter(Classifier):

    '''
    Classifier -> The actual python class of the model
                  used, so self.classifier() creates
                  an instance of that class
    '''
    def __init__(self, classifier, approach=Approach.ONE_VS_ALL):
        super().__init__(classifier.N, 1, 'BinaryClassifierAugmenter of '+ classifier.name, _type=255)
        self.classifier = classifier 
        self.approach = approach

    def fit(self, X, Y, epochs=100):
        self.classes = np.unique(Y)
        self.M = len(self.classes)
        classifiers = [self.classifier]+[deepcopy(self.classifier) for _ in range(self.M-1)]

        if self.approach == Approach.ONE_VS_ALL:
            # i-th index -> labels for i-th classifier
            classifiers_data = []
            for cl in self.classes:
                labels = np.zeros(len(Y))
                labels[Y==cl] = 1
                classifiers_data.append(labels)

            accuracies = []

            for i, classifier in enumerate(classifiers):
                classifier.fit(X, classifiers_data[i], epochs=epochs)
                accuracies.append(classifier.accuracy(X, classifiers_data[i]))
            
            # the accuracy is gonna be used as weights to determine which classifier is
            # more likely to be right if both predict the class of a sample to be theirs
            self.accuracies = np.array(accuracies)
        elif self.approach == Approach.CLASS_IS_REACHED:
            raise NotImplementedError("Method not yet implemented!")

        self.classifiers = classifiers

    def _predict(self, x):
        if self.approach == Approach.ONE_VS_ALL:
            predictions = np.array([classifier(x) for classifier in self.classifiers]) # [0,1,1,0,1]
            predicted_class = np.argmax(self.accuracies*predictions) # arg max([0, acc1, acc2, 0, acc4])
            return self.classes[predicted_class]

    def _save(self, file: h5py.File) -> None:
        for i, classifier in enumerate(self.classifiers):
            clf_group = file.create_group(f'classifier_{i}')
            classifier.save_in_file(clf_group)
        file.create_dataset('accuracies', self.accuracies.shape, np.float32, self.accuracies, compression="gzip")
    
    def _load(self, file: h5py.File) -> None:
        self.classifiers = []
        curr_classifier_idx = 0
        while f'classifier_{curr_classifier_idx}' in file.keys():
            # load the classifier from the group
            classifier = Classifier.load_classifier(file[f'classifier_{curr_classifier_idx}'])
            self.classifiers.append(classifier)
            curr_classifier_idx += 1
        self.accuracies = np.array(file['accuracies'])