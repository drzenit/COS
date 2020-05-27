import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data, io
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd


# open the camera image
#img = Image.open("data\Powdery_mildew\\1.jpg")

#grayscale = img.convert('L')
#image = np.asarray(grayscale)

image = io.imread("data\Powdery_mildew\\1.jpg", as_gray=True)
image = np.round(image * 255)
image = image.astype(int)




glcm = greycomatrix(image, distances=[2], angles=[0], levels=256)
gr = greycoprops(glcm, "homogeneity")

print(gr)

# compute some GLCM properties each patch
xs = []
ys = []


print(xs)
print(ys)
# create the figure
