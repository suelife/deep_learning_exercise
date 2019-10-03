import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import *
import numpy as np
import pickle

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    return x_test, y_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    l1 = np.dot(x, W1) + b1
    a1 = sigmoid(l1)
    l2 = np.dot(a1, W2) + b2
    a2 = sigmoid(l2)
    l3 = np.dot(a2, W3) + b3
    y = softmax(l3)

    return y


x_test, y_test = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p = np.argmax(y)
    if p == y_test[i]:
        accuracy_cnt += 1
print('Accuracy: {}'.format(str(float(accuracy_cnt / len(x_test)))))