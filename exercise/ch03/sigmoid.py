import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv = False):
    if deriv == True:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)
    plt.show()