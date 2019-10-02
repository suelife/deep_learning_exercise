import numpy as np
from andGate import AND
from nandGate import NAND
from orGate import OR

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)

    return y

if __name__ == '__main__':
    x_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for xs in x_list:
        y = XOR(xs[0], xs[1])
        print('{} -> {}'.format(str(xs), str(y)))