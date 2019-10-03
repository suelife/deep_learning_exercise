import os, sys
sys.path.append(os.pardir)
import numpy as np
from common.functions import *

def init_network():
    network = {}
    network['W1'] = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    network['b1'] = [0.1, 0.2, 0.3]
    network['W2'] = [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
    network['b2'] = [0.1, 0.2]
    network['W3'] = [[0.1, 0.3], [0.2, 0.4]]
    network['b3'] = [0.1, 0.2]
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    l1 = np.dot(x, W1) + b1
    a1 = sigmoid(l1)
    l2 = np.dot(a1, W2) + b2
    a2 = sigmoid(l2)
    l3 = np.dot(a2, W3) + b3
    y = identity_function(l3)

    return y

x = np.array([1.0, 0.5])
network = init_network()
y = forward(network, x)
print(y)