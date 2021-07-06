import numpy as np
from networks.errors import *
from networks.optimizers import *
from networks.std.network import Model
from networks.std.layer import Layer

import tensorflow as tf
from tensorflow.keras import layers

print("\n"*10)

SAMPLE_SIZE = 300
X = np.random.uniform(low=-5, high=5, size=(SAMPLE_SIZE, 3))
target_weights = np.random.uniform(low=-10, high=1, size=(3,4))
Y = np.dot(X, target_weights)

model = Model([Layer(3), Layer(7,activation=ReLu), Layer(4)])
model.compile(MSE(), SGD(lr=0.001))

tf_model = tf.keras.Sequential(
	tf.keras.Input(shape=(3,)),
	layers.Dense(5, activation='relu'),
	layers.Dense(4)
	)
tf_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
				loss='mse',
				metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(X,Y, epochs=1000)
print("\n"*10)
tf_model.fit(X,Y, epochs=1000)

'''

momentum = Model([Layer(3), Layer(4)])
momentum.compile(MSE(), Momentum(lr=0.1))

adagrad = Model([Layer(3), Layer(4)])
adagrad.compile(MSE(), Adagrad(lr=0.3))

adadelta = Model([Layer(3), Layer(4)])
adadelta.compile(MSE(), Adadelta(lr=0.03))

adam = Model([Layer(3), Layer(4)])
adam.compile(MSE(), Adam(lr=0.3))


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
'''