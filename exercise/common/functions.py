import numpy as np
import matplotlib.pyplot as plt


# ch03
def step_function(x):
    return np.array(x > 0).astype(np.int) # or np.array(x>0, dtype=np.int) 

def sigmoid(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    total_exp = np.sum(exp_x)
    return exp_x / total_exp


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    step_y = step_function(x)
    sig_y = sigmoid(x)
    relu_y = relu(x)
    
    plt.subplot(1, 3, 1, figsize=(15,15))
    plt.plot(x, step_y)
    plt.title('Step Function')

    plt.subplot(1, 3, 2)
    plt.plot(x, sig_y)
    plt.title('Sigmoid')

    plt.subplot(1, 3, 3)
    plt.plot(x, relu_y)
    plt.title('ReLU')

    plt.show()
    