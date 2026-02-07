import numpy as np
from zernike import RZern
from PIL import Image

# Load your image
img = Image.open('lena.png').convert('L')  # Convert to grayscale
image_data = np.array(img, dtype=float)
# Trim image to square
L, K = image_data.shape
min_dim = min(L, K)
image_data = image_data[:min_dim, :min_dim]
# Create Zernike object (up to radial order 6)
zern = RZern(16)

# Set up grid matching your image size
L, K = image_data.shape  # height, width
dd = np.linspace(-1.0, 1.0, K)
dy = np.linspace(-1.0, 1.0, L)
xx, yy = np.meshgrid(dd, dy)

# Create cartesian grid basis
zern.make_cart_grid(xx, yy, unit_circle=True)

# Compute Zernike moments (coefficients) from image
moments, res, rnk, sv = zern.fit_cart_grid(image_data)
print(moments.shape)

moments_normalized = (moments / moments[0]) * 50  # Normalize by the first moment (Z_0_0)
half_moments = moments.copy()
half_moments[half_moments.shape[0]//2:] = 0  # Keep only the first moment, zero
# zern1 = RZern(1)
# zern1.make_cart_grid(xx, yy, unit_circle=True)

newImg_normalized = zern.eval_grid(moments_normalized, matrix=True)
newImg = zern.eval_grid(moments, matrix=True)
halfImg = zern.eval_grid(half_moments, matrix=True)

import matplotlib.pyplot as plt
finalImage = image_data + newImg
finalImage_normalized = image_data + newImg_normalized
finalImage_half = image_data + halfImg

# WATER_MARK_ORIG = np.random.randint(0, 2, size=128)
WATER_MARK_ORIG = np.array([1,0,1,0,1,0,1,1,1,1,0,0,0,0])
WATER_MARK = np.pad(WATER_MARK_ORIG, (0, moments.shape[0] - WATER_MARK_ORIG.shape[0]), mode='constant')
print("Watermark bits:", WATER_MARK)

delta = 4
moments_normalized_abs = np.abs(moments_normalized)
moments_normalized_abs_frac, moments_normalized_abs_dec = np.modf(moments_normalized_abs)
moments_q = (moments_normalized_abs_dec // delta) * delta
moments_q = moments_q + moments_normalized_abs_frac + 0.25 * delta + WATER_MARK * 0.5


z_w = (moments_q / moments_normalized_abs) * moments_normalized
moments_for_watermark = z_w - moments

irw = zern.eval_grid(moments_for_watermark, matrix=True)
final_watermarked_image = image_data + irw


plt.figure(figsize=(30, 5))
plt.subplot(1, 6, 1)
plt.imshow(image_data, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 6, 2)
plt.imshow(newImg, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.subplot(1, 6, 3)
plt.imshow(finalImage, cmap='gray')
plt.title('Combined Image')
plt.axis('off')
plt.subplot(1, 6, 4)
plt.imshow(finalImage_normalized, cmap='gray')
plt.title('Reconstructed Image (Normalized)')
plt.axis('off')
plt.subplot(1, 6, 5)
plt.imshow(finalImage_half, cmap='gray')
plt.title('Combined Image (Half Moments)')
plt.axis('off')
plt.subplot(1, 6, 6)
plt.imshow(final_watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('zernike/res.png')
plt.show()

print("z_w")
print(np.array2string(z_w, formatter={'float_kind': lambda x: f'{x:.5f}'}))
print("original moments")
print(np.array2string(moments, formatter={'float_kind': lambda x: f'{x:.5f}'}))