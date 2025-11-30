import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Function to show images
def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load cover image (grayscale)
cover_img = cv2.imread("cover.jpg", cv2.IMREAD_GRAYSCALE)
cover_img = cv2.resize(cover_img, (256, 256))

# Load watermark image (binary)
watermark = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)
watermark = cv2.resize(watermark, (128, 128))
_, watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)

# Perform DWT
coeffs = pywt.dwt2(cover_img, 'haar')
LL, (LH, HL, HH) = coeffs

# Embed watermark into HL band
alpha = 0.05  # strength factor
HL_watermarked = HL + alpha * watermark

# Reconstruct image
watermarked_img = pywt.idwt2((LL, (LH, HL_watermarked, HH)), 'haar')

# Normalize to [0,255] and convert to uint8
watermarked_img_norm = np.clip(watermarked_img, 0, 255).astype(np.uint8)

# Show results
show_image(cover_img, "Original Cover Image")
show_image(watermark, "Watermark")
show_image(watermarked_img_norm, "Watermarked Image")

# Save watermarked image
cv2.imwrite("watermarked.png", watermarked_img_norm)
print("✅ Watermarked image saved as 'watermarked.png'")
