import numpy as np

# TODO: finish the stupid ass gradients


'''
grad() returns ∂Hi/∂y where Hi is the inside of the error function
it's the part that is commmon in all of the derivative, caused/
which shows up because of the chain rule
# https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
'''
class ErrorFunction:

	def __init__(self, name):
		self.name = name

class MSE(ErrorFunction):

	def __init__(self):
		super().__init__('Mean Squared Error')

	def __call__(self, Y, predictions):
		return np.sum((predictions-Y)**2)/len(Y)

	def grad(self, Y, predictions):
		return 2*(predictions-Y)/len(Y)

# Mean Absolute Error (or Huber loss)
class MAE(ErrorFunction):

	def __init__(self):
		super().__init__('Mean Absolute Error')

	def __call__(self, Y, predictions):
		return np.sum(np.abs(predictions-Y))/len(Y)

	def grad(self, Y, predictions):
		return np.sign(predictions-Y)/len(Y)

# Mean Bias Error (not recommended)
class MBE(ErrorFunction):

	def __init__(self):
		super().__init__('Mean Bias Error')

	def __call__(self, Y, predictions):
		return np.sum(predictions-Y)/len(Y)

	def grad(self, Y, predictions):
		return np.ones(Y.shape)

class CrossEntropy(ErrorFunction):

	def __init__(self):
		super().__init__('Cross Entropy')

	def __call__(self, Y, predictions):
		return -np.sum(Y*np.log(predictions)+(1-Y)*np.log(1-predictions))

	def grad(self, Y, predictions):
		return (predictions-Y)/(len(Y)*(1-predictions)*predictions)

class CategoricalCrossEntropy(ErrorFunction):

	def __init__(self):
		super().__init__('Categorical Cross Entropy')

	def __call__(self, Y, predictions):
		return -np.sum(Y*np.log(predictions))

	def grad(self, Y, predictions):
		return -Y/predictions

class KullbackLeiblerDivergence(ErrorFunction):

	def __init__(self):
		super().__init__('Kullback-Leibler divergence')

	def __call__(self, Y, predictions):
		return np.sum(Y*np.log(Y/predictions))

	def grad(self, Y, predictions):
		return -Y/predictions

class Exponential(ErrorFunction):

	def __init__(self, t=0.1):
		super().__init__('Exponential')
		self.t = t

	def __call__(self, Y, predictions):
		return self.t*np.exp(np.sum((Y-predictions)**2)/self.t)

	def grad(self, Y, predictions):
		return 2/self.t*self.__call__(Y, predictions)*(predictions-Y)

class HellingerDistance(ErrorFunction):

	def __init__(self):
		super().__init__('Hellinger Distance')

	def __call__(self, Y, predictions):
		return np.sum((np.sqrt(Y)-np.sqrt(predictions))**2)/np.sqrt(2)

	def grad(self, Y, predictions):
		return (np.sqrt(predictions)-np.sqrt(Y))/(np.sqrt(2*predictions))
