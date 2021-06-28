import numpy as np

class KNN:

	def __init__(self, neighbors_amt=3):
		self.neighbors_amt = neighbors_amt

	def predict(self, x):
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
		self.X = X
		self.Y = Y