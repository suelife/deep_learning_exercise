import os, sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import matplotlib.pyplot as plt

def img_show_pil(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def img_show_plt(img):
    plt.imshow(img)
    plt.show()

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = y_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

# img_show_pil(img)
img_show_plt(img)
