import os, sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=False)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

def img_show_pil(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def img_show_plt(img):
    plt.imshow(img, cmap='gray')
    plt.show()

img = x_train[0]
label = y_train[0]

print(img.shape)
img = np.reshape(img, (28, 28))
print(img.shape)

print("label: {}".format(label))
img_show_plt(img)