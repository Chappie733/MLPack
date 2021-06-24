import numpy as np

#In the following functions args[0] is actually kwargs

def parser(func):

	def wrapper(x, deriv=False, **kwargs):
		if not isinstance(deriv, bool):
			raise TypeError("Expected the parameter \'deriv\' to be a boolean, but received {type} instead!".format(type=type(deriv)))
		elif not isinstance(x, np.ndarray) and not isinstance(x, Number):
			raise TypeError("Expected the parameter \'x\' to be a numpy array or a number, but received {type} instead!".format(type=type(x)))
		return func(x, deriv, kwargs)

	return wrapper

@parser
def linear(x, deriv=False, *args):
	return x if not deriv else np.ones(x.shape)

@parser
def sigmoid(x, deriv=False, *args):
	return 1/(1+np.exp(-x)) if not deriv else sigmoid(x)*(1-sigmoid(x))

@parser
def tanh(x, deriv=False, *args):
	return np.tanh(x) if not deriv else 1-np.tanh(x)**2

@parser
def ReLu(x, deriv=False, *args):
	return (np.abs(x)+x)/2 if not deriv else (np.sign(x)+1)/2

@parser
def ELU(x, deriv=False, *args):
	alpha = 1 if 'alpha' not in args[0] else args[0]['alpha']
	return np.where(x>0, x, alpha*(np.exp(x)-1)) if not deriv else np.where(x>0, 1, alpha*np.exp(x))

@parser
def LeakyReLu(x, deriv=False, *args):
	alpha = 1 if 'alpha' not in args[0] else args[0]['alpha']
	return np.where(x>0, x, alpha*x) if not deriv else np.where(x>0, 1, alpha)

@parser
def atan(x, deriv=False, *args):
	return np.arctan(x) if not deriv else 1/(x**2+1)

@parser
def BentIdentity(x, deriv=False, *args):
	return x+(np.sqrt(x**2+1)-1)/2 if not deriv else x/(2*np.sqrt(x**2+1))+1

@parser
def BipolarSigmoid(x, deriv=False, *args):
	return (1-np.exp(-x))/(1+np.exp(x)) if not deriv else (2+np.exp(-x)-np.exp(x))/((1+np.exp(x))**2)

@parser
def gaussian(x, deriv=False, *args):
	return np.exp(-x**2) if not deriv else -2*x*np.exp(-x**2)

@parser
def hardtanh(x, deriv=False, *args):
	return np.where(np.abs(x)>1, np.sign(x), x)	 if not deriv else np.where(np.abs(x)>1, 0, 1)

@parser
def InverseSqrt(x, deriv=False, *args):
	alpha = 0.1 if 'alpha' not in args[0] else args[0]['alpha']
	return x/(np.sqrt(1+alpha*x**2)) if not deriv else 1/(1+a*x**2)**(3/2)

@parser
def LeCunTanh(x, deriv=False, *args):
	alpha = 1.7159 if 'alpha' not in args[0] else args[0]['alpha']
	return alpha*np.tanh(2*x/3) if not deriv else 2*alpha/(3*np.cosh(2*x/3)**2)

@parser
def LogLog(x, deriv=False, *args):
	return 1-np.exp(-np.exp(x)) if not deriv else np.exp(x)*(LogLog(x)-1)

@parser
def LogSigmoid(x, deriv=False, *args):
	return np.log(sigmoid(x)) if not deriv else 1-sigmoid(x)

@parser
def SELU(x, deriv=False, *args):
	alpha = 1.67326 if 'alpha' not in args[0] else args[0]['alpha']
	beta = 1.0507 if 'beta' not in args[0] else args[0]['beta']
	return beta*np.where(x>0, x, alpha*(np.exp(x-1))) if not deriv else beta*np.where(x>0, 1, alpha*np.exp(x))

@parser
def sinc(x, deriv=False, *args):
	return np.where(x!=0, np.sin(x)/x, 1) if not deriv else np.where(x!=0, np.cos(x)/x-np.sin(x)/(x**2), 0)

@parser
def swish(x, deriv=False, *args):
	return x/(1+np.exp(-x)) if not deriv else np.exp(x)*(x+np.exp(x)+1)/(np.exp(x)+1)**2

@parser
def softsign(x, deriv=False, *args):
	return x/(1+np.abs(x)) if not deriv else 1/(1+np.abs(x))**2

@parser
def softplus(x, deriv=False, *args):
	return np.log(1+np.exp(x)) if not deriv else np.exp(x)/(1+np.exp(x))

'''
TO IMPLEMENT:
softmax

'''
