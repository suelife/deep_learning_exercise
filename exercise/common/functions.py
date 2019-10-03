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


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    step_y = step_function(x)
    sig_y = sigmoid(x)
    
    plt.subplot(1, 2, 1)
    plt.plot(x, step_y)
    plt.title('step_function')

    plt.subplot(1, 2, 2)
    plt.plot(x, sig_y)
    plt.title('sigmoid')

    plt.show()
    