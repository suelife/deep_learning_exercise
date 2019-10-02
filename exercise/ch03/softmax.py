import numpy as np

def softmax(x):
    x_max = np.max(x)
    x_exp = np.exp(x - x_max) # minus the max value from x, to prevent overflow
    x_sum_exp = np.sum(x_exp)

    return x_exp / x_sum_exp

if __name__ == '__main__':
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
