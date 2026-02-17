import numpy as np
from zernike import RZern
from PIL import Image
import matplotlib.pyplot as plt
from zernike_helper import fill_moments_array, generate_zernike_pairs, get_moments_for_pairs, embed_watermark_in_moments, get_water_mark_from_moments

N = 8
T = 120
delta = 10

pairs, cong_pairs = generate_zernike_pairs(N)
WATER_MARK_ORIG = np.array([1,0,1,0,1,0,1,1,1,1,0,0,1,1])
pairs = pairs[:len(WATER_MARK_ORIG)]
cong_pairs = cong_pairs[:len(WATER_MARK_ORIG)]

# Load your image
img = Image.open('lena.png').convert('L')  # Convert to grayscale
image_data = np.array(img, dtype=float)
# image_data = image_data[200:400, 200:400]
# Trim image to square
L, K = image_data.shape
print("Original image shape:", image_data.shape)
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
z_0_0 = moments[0]
print("Zernike moments shape:", moments.shape)
print("z_0_0:", z_0_0)
zern_moments = get_moments_for_pairs(moments, pairs, zern)
print("Zernike moments for selected pairs:")
for (n, m), val in zip(pairs, zern_moments):
    print(f"Moment (n={n}, m={m}): {val:.5f}")

zern_cong_moments = get_moments_for_pairs(moments, cong_pairs, zern)
print("\nConjugate Zernike moments for selected pairs:")
for (n, m), val in zip(cong_pairs, zern_cong_moments):
    print(f"Moment (n={n}, m={m}): {val:.5f}")

embedded_moments = embed_watermark_in_moments(zern_moments, z_0_0, WATER_MARK_ORIG, T=T, delta=delta)
embedded_moments_cong = embed_watermark_in_moments(zern_cong_moments, z_0_0, WATER_MARK_ORIG, T=T, delta=delta)

embedded_moments = embedded_moments - zern_moments
embedded_moments_cong = embedded_moments_cong - zern_cong_moments

print("\nEmbedded Zernike moments for selected pairs:")
for (n, m), val in zip(pairs, embedded_moments):
    print(f"Moment (n={n}, m={m}): {val:.5f}")
print("\nEmbedded Conjugate Zernike moments for selected pairs:")
for (n, m), val in zip(cong_pairs, embedded_moments_cong):
    print(f"Moment (n={n}, m={m}): {val:.5f}")

embeded_image_moments = np.zeros(len(moments))
fill_moments_array(embeded_image_moments, pairs, embedded_moments, zern)
fill_moments_array(embeded_image_moments, cong_pairs, embedded_moments_cong, zern)
reconstructed_image = zern.eval_grid(embeded_image_moments, matrix=True)
#print("shape of reconstruct image:", reconstructed_image.shape)
#array_string = np.array2string(reconstructed_image, threshold=np.inf)
#print(array_string)
reconstructed_image[np.isnan(reconstructed_image)] = 0

final_image = image_data + reconstructed_image
plt.imshow(image_data, cmap='gray')
plt.title("Image with Embedded Watermark in Zernike Moments")
plt.axis('off')
plt.show()

received_image = final_image  # Simulate receiving the image
#received_image = received_image[:min_dim, :min_dim]
received_moments, _, _, _ = zern.fit_cart_grid(received_image)
z_0_0 = received_moments[0]
print("z_0_0:", z_0_0)
received_zern_moments = get_moments_for_pairs(received_moments, pairs, zern)
received_zern_cong_moments = get_moments_for_pairs(received_moments, cong_pairs, zern)
recovered_watermark = get_water_mark_from_moments(received_zern_moments, z_0_0, T=T, delta=delta)
recovered_watermark_cong = get_water_mark_from_moments(received_zern_cong_moments, z_0_0, T=T, delta=delta)
print("Recovered Watermark Bits:          ", recovered_watermark)
print("Recovered Conjugate Watermark Bits:", recovered_watermark_cong)
print("Original Watermark Bits:           ", WATER_MARK_ORIG)
