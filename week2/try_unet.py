from Unet import Unet
from scipy.misc import imread, imsave
import numpy as np

img = imread('../datasets/portraits/imgs/42.jpg')

x = np.zeros((1, 800, 600, 3), dtype='float32')
x[0] = img

model = Unet(2, 'adam', 600, 800)
model.load_weights('weights/unet.hdf5')
y = model.predict(x)

imsave('test.jpg', y[0, :, :, 0] * 255)
