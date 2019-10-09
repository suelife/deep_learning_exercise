import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    '''
    微分中的h趨近於0，但數值微分中的h並非如此
    因此數值微分所求解為(x+h)與x之間的斜率，而非x的斜率
    為了改善誤差所造成的情況，計算(x+h)與(x-h)的函數f差分
    藉此減少之間的誤差
    '''
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y_f = f(x)
    d_x = d*x
    print(y_f)
    print(d_x)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y1 = function_1(x)
plt.plot(x, y1)
plt.xlabel('x')
plt.ylabel('f(x)')

tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y2)

plt.show()