import numpy as np
from classification.classifier import Classifier

class KNN(Classifier):

	def __init__(self, name, neighbors_amt=3):
		super().__init__(1, 2, name)
		self.neighbors_amt = neighbors_amt

	def _predict(self, x, *args, **kwargs):
		res = np.sum((x-self.X)**2, axis=1)
		vals = np.sort(res)[:self.neighbors_amt]
		indices = [np.where(res==vals[i])[0][0] for i in range(len(vals))]
		classes = np.array([self.Y[i] for i in indices])
		values, counts = np.unique(classes, return_counts=True)
		ind = np.argmax(counts)
		if np.array_equal(counts-np.mean(counts), np.zeros(shape=counts.shape)):
			return classes[0] # if all of the elements appear the same amount of times, return the closest to the input to predict
		return classes[ind]

	def fit(self, X, Y):
		self.N = X.shape[1]
		self.X = X
		self.Y = Y
	
	def _save(self, file):
		try:
			file.create_dataset('X', self.X.shape, self.X.dtype, self.X, compression="gzip")
			file.create_dataset('Y', self.Y.shape, self.Y.dtype, self.Y, compression="gzip")
		except AttributeError:
			raise Exception("You need to have called the fit method before you can save the model!")

	def _load(self, file):
		self.X = np.array(file['X'])
		self.Y = np.array(file['Y'])