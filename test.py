from PIL import Image
import numpy as np
from my_zernike import zernike_helper
import matplotlib.pyplot as plt


img = Image.open('cat.png').convert('L')  # Convert to grayscale
image_data = np.array(img, dtype=float)
L, K = image_data.shape
print("Original image shape:", image_data.shape)
min_dim = min(L, K)
image_data = image_data[:min_dim, :min_dim]
WATER_MARK_ORIG = np.array([1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,1])

final_image = zernike_helper.embed_watermark(image_data, WATER_MARK_ORIG, 8)
rotated_img = np.rot90(final_image, -1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_data, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(final_image, cmap='gray')
plt.title("Image with Embedded Watermark in Zernike Moments")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(rotated_img, cmap='gray')
plt.title("Image with Embedded Watermark in Zernike Moments rotated")
plt.axis('off')
plt.show()

recovered_watermark, recovered_watermark_cong = zernike_helper.recover_matermark(final_image)
print("Result from received rotated image:")
print("Original Watermark Bits:           ", WATER_MARK_ORIG)
print("Recovered Watermark Bits:          ", recovered_watermark)
print("Recovered Conjugate Watermark Bits:", recovered_watermark_cong)

recovered_watermark, recovered_watermark_cong = zernike_helper.recover_matermark(rotated_img)
print("Result from received rotated image:")
print("Original Watermark Bits:           ", WATER_MARK_ORIG)
print("Recovered Watermark Bits:          ", recovered_watermark)
print("Recovered Conjugate Watermark Bits:", recovered_watermark_cong)

