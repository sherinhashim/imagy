import numpy as np
from zernike import RZern
from PIL import Image
from zernike_helper import generate_zernike_pairs, get_moments_for_pairs

N = 8
pairs, cong_pairs = generate_zernike_pairs(N)
# Load your image
img = Image.open('lena.png').convert('L')  # Convert to grayscale
image_data = np.array(img, dtype=float)
# Trim image to square
L, K = image_data.shape
min_dim = min(L, K)
image_data = image_data[:min_dim, :min_dim]
# Create Zernike object (up to radial order 6)
zern = RZern(N)

# Set up grid matching your image size
L, K = image_data.shape  # height, width
dd = np.linspace(-1.0, 1.0, K)
dy = np.linspace(-1.0, 1.0, L)
xx, yy = np.meshgrid(dd, dy)

# Create cartesian grid basis
zern.make_cart_grid(xx, yy, unit_circle=True)


# Compute Zernike moments (coefficients) from image
moments, res, rnk, sv = zern.fit_cart_grid(image_data)
print("Zernike moments shape:", moments.shape)
zern_moments = get_moments_for_pairs(moments, pairs, zern)
print("Zernike moments for selected pairs:")
for (n, m), val in zip(pairs, zern_moments):
    print(f"Moment (n={n}, m={m}): {val:.5f}")

zern_cong_moments = get_moments_for_pairs(moments, cong_pairs, zern)
print("\nConjugate Zernike moments for selected pairs:")
for (n, m), val in zip(cong_pairs, zern_cong_moments):
    print(f"Moment (n={n}, m={m}): {val:.5f}")
