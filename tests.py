import numpy as np
from networks.errors import *
from networks.optimizers import *
from networks.std.network import Model
from networks.std.layer import Layer

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
