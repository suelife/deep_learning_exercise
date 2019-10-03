import os, sys
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist

# Activation Function
from softmax import softmax
from sigmoid import sigmoid

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
    y = sigmoid(l3)

    return y

x_test, y_test = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y = predict(network, x_batch)
    p = np.argmax(y, axis=1)
    accuracy_cnt += np.sum(p == y_test[i:i+batch_size])
    if i % 2000 == 0:
        print('Accuracy: ' + str(float(accuracy_cnt) / len(x_test)))

print('Final Accuracy: ' + str(float(accuracy_cnt) / len(x_test)))