import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-04
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x

    print(lambda t: d*t + y)

    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

tf = tangent_line(function_1, 5)
print(tf)
y2 = tf(x)

# plt.plot(x, y)
# plt.plot(x, y2)
# plt.xlabel('x')
# plt.ylabel('f(x)')

# plt.show()