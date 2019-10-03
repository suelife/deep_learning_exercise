import os, sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import *

x = np.arange(-5.0, 5.0, 0.1)
y_step = step_function(x)
y_sig = sigmoid(x)

plt.plot(x, y_step)
plt.plot(x, y_sig, '--')
plt.ylim(-0.1, 1.1)
plt.show()