import mahotas
import mahotas.features
from skimage import io, color
import numpy as np

# Read your image (grayscale or color)
img = io.imread('Lena.png')

# Convert to grayscale (Zernike needs single-channel)
if img.ndim == 3:
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

# Define radius (half of image size usually)
radius = min(img_gray.shape) // 2

# Compute Zernike moments up to order 8
zernike_moments = mahotas.features.zernike_moments(img_gray, radius, degree=3)

print("Number of Zernike moments:", len(zernike_moments))
print("First 10 Zernike moments:\n", zernike_moments[:10])
