import numpy as np
from networks.errors import *
from networks.optimizers import *
from networks.activations import ReLu
from networks.std.network import Model
from networks.std.layer import Layer
from unsupervised.KMC import KMC
from unsupervised.linear_model import LinearModel

import os
import sys
import matplotlib.pyplot as plt
import time

if len(sys.argv) == 1:
	sys.argv.append('')

dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, 'test_save.h5')


if sys.argv[1] == 'model_save_test':
	model = Model([Layer(3), Layer(4, activation=ReLu)])
	model.compile(MSE(), Momentum(lr=0.001))
	print(model)
	model.save('test_save')

elif sys.argv[1] == 'model_load_test':
	model = Model([])
	model.load('test_save')
	print(model)

elif sys.argv[1] == 'KMC_vis':
	X = np.array([[3,0],
				  [2,0],
				  [3,0],
				  [0,3],
				  [0,2],
				  [0.5,2.5],
				  [-3,-3],
				  [-2,-3],
				  [-3,-2]])

	model = KMC(2, 3, lr=0.01)

	fig = plt.figure()

	def animate():
		for i in range(20):
			ax = fig.add_subplot(1, 1, 1)
			ax.scatter(X[:3,0], X[:3,1], color='red')
			ax.scatter(X[3:6,0], X[3:6,1], color='blue')
			ax.scatter(X[6:,0], X[6:,1], color='black')
			for j in range(model.K):
				ax.scatter([model.weights[j][0]], [model.weights[j][1]], color='#009933', s=40, marker='x')
			model.fit(X,epochs=5, mode='descent')
			fig.canvas.draw()
			time.sleep(0.5)

	fig.suptitle('Input Space', fontweight ="bold")
	fig.canvas.manager.window.after(20, animate)
	plt.show()

elif sys.argv[1] == 'uLinear_vis':
	X = np.array([[3,0],
				  [2,0],
				  [3,0],
				  [0,3],
				  [0,2],
				  [0.5,2.5],
				  [-3,-3],
				  [-2,-3],
				  [-3,-2]])

	model = LinearModel(2, 3, lr=0.01)
	   
	fig = plt.figure()

	def animate():
		for i in range(20):
			ax = fig.add_subplot(1, 1, 1)
			ax.scatter(X[:3,0], X[:3,1], color='red')
			ax.scatter(X[3:6,0], X[3:6,1], color='blue')
			ax.scatter(X[6:,0], X[6:,1], color='black')
			for j in range(model.M):
				ax.scatter([model.weights[j][0]], [model.weights[j][1]], color='#009933', s=40, marker='x')
			model.fit(X,epochs=5)
			fig.canvas.draw()
			time.sleep(0.5)

	fig.suptitle('Input Space', fontweight ="bold")
	fig.canvas.manager.window.after(20, animate)
	plt.show()

elif sys.argv[1] == 'perceptron_tests':
	NUM_SAMPLES = 1000
	NUM_FEATURES = 5

	inputs = np.random.uniform(low=-1, high=1, size=(NUM_SAMPLES, NUM_FEATURES))
	target_weights = np.random.uniform(low=-3, high=3, size=(NUM_FEATURES,))
	target_bias = np.random.uniform(low=-3, high=3)
	outputs = np.sign(np.dot(inputs, target_weights)+target_bias)

	model = Perceptron(NUM_FEATURES)
	model.fit(inputs, outputs, method='pocket', steps=1000)
	acc = len(np.where(model.predict(inputs) == outputs)[0])/NUM_SAMPLES
	print(f"Pocket algorithm accuracy: {acc}")
	del model

	model = Perceptron(NUM_FEATURES)
	model.fit(inputs, outputs, method='pla', steps=1000)
	acc = len(np.where(model.predict(inputs) == outputs)[0])/NUM_SAMPLES
	print(f"Perceptron learning algorithm accuracy (pla): {acc}")
	del model

	model = Perceptron(NUM_FEATURES)
	model.fit(inputs, outputs, method='rosenblatt', steps=1000)
	acc = len(np.where(model.predict(inputs) == outputs)[0])/NUM_SAMPLES
	print(f"Rosenblatt rule accuracy: {acc}")


elif sys.argv[1] == 'optimizers_comparison':
	SAMPLE_SIZE = 300
	X = np.random.uniform(low=-5, high=5, size=(SAMPLE_SIZE, 3))
	target_weights = np.random.uniform(low=-10, high=1, size=(3,4))
	Y = np.dot(X, target_weights)

	sgd = Model([Layer(3), Layer(4)])
	sgd.compile(MSE(), SGD(lr=0.01))

	momentum = Model([Layer(3), Layer(4)])
	momentum.compile(MSE(), Momentum(lr=0.01))

	adagrad = Model([Layer(3), Layer(4)])
	adagrad.compile(MSE(), Adagrad(lr=0.01))

	adadelta = Model([Layer(3), Layer(4)])
	adadelta.compile(MSE(), Adadelta(lr=0.01))

	adam = Model([Layer(3), Layer(4)])
	adam.compile(MSE(), Adam(lr=0.01))

	print("\n"*4)
	print("SGD", end='\n'*2)
	sgd_errors = sgd.fit(X,Y, return_errors=True)

	print("\n"*4)
	print("Momentum", end='\n'*2)
	momentum_errors = momentum.fit(X,Y, return_errors=True)

	print("\n"*4)
	print("Adagrad", end='\n'*2)
	adagrad_errors = adagrad.fit(X,Y, return_errors=True)

	print("\n"*4)
	print("Adadelta", end='\n'*2)
	adadelta_errors = adadelta.fit(X,Y, return_errors=True)

	print("\n"*4)
	print("Adam", end='\n'*2)
	adam_errors = adam.fit(X,Y, return_errors=True)

	import matplotlib.pyplot as plt

	plt.plot([epoch for epoch in range(10, 76)], sgd_errors[9:])
	plt.plot([epoch for epoch in range(10, 76)], momentum_errors[9:])
	plt.plot([epoch for epoch in range(10, 76)], adagrad_errors[9:])
	plt.plot([epoch for epoch in range(10, 76)], adadelta_errors[9:])
	plt.plot([epoch for epoch in range(10, 76)], adam_errors[9:])

	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.show()