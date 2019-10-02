import numpy as np

def NAND(x1, x2):
    x = np.array([x1, x2])
    W = np.array([-0.5, -0.5])
    b = 0.7 
    tmp = np.sum(x*W) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    x_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for xs in x_list:
        y = NAND(xs[0], xs[1])
        print('{} -> {}'.format(str(xs), str(y)))
