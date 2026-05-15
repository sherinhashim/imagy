from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import zernike
import helper

DELTA     = 20       # QIM quantization step Δ
T         = 1000     # amplification factor T  (paper uses 1000)
nm_pairs   = [(1,1), (2,2), (3,1), (3,3), (4,2), (5,1), (5,3)]
watermark  = [1, 0, 1, 1, 0, 1, 0]

full_img = Image.open('s_lena.jpeg').convert("YCbCr")
img_arr = np.array(full_img)
y_img = img_arr[:,:,0]
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(y_img, cmap='gray')
plt.title("Original Image")

selected_points = helper.get_circles(full_img)

for x, y, r, desc, score in selected_points:
    print(f"Point: ({x:.1f}, {y:.1f}), Radius: {r:.1f}, Score: {score:.4f}")
    r = math.floor(r)
    x1 = x-r
    x2 = x+r
    y1 = y-r
    y2 = y+r
    square = y_img[y1:y2, x1:x2]
    Irw, Iw, cache = zernike.compute_Irw(square, nm_pairs, watermark, delta=DELTA, T=T)
    y_img[y1:y2, x1:x2] = Iw

plt.subplot(1, 3, 2)
plt.imshow(y_img, cmap='gray')
plt.title("embeded image");
plt.show()

img_arr[:,:,0] = y_img
new_img = Image.fromarray(img_arr, mode="YCbCr")
new_img = new_img.convert("RGB")
new_img.save("output.jpg", "JPEG")


for x, y, r, desc, score in selected_points:
    print(f"Point: ({x:.1f}, {y:.1f}), Radius: {r:.1f}, Score: {score:.4f}")
    r = math.floor(r)
    x1 = x-r
    x2 = x+r
    y1 = y-r
    y2 = y+r
    square = y_img[y1:y2, x1:x2]
    extracted = zernike.extract_watermark(square, nm_pairs, delta=DELTA, T=T)
    print("Result from received image:")
    print("Original Watermark Bits:           ", watermark)
    print("Recovered Watermark Bits:          ", extracted)

# rotated_img = np.rot90(final_image, -1)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(image_data, cmap='gray')
# plt.title("Original Image")
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.imshow(final_image, cmap='gray')
# plt.title("Image with Embedded Watermark in Zernike Moments")
# plt.axis('off')
# plt.subplot(1, 3, 3)
# plt.imshow(rotated_img, cmap='gray')
# plt.title("Image with Embedded Watermark in Zernike Moments rotated")
# plt.axis('off')
# plt.show()


# recovered_watermark, recovered_watermark_cong = zernike_helper.recover_matermark(rotated_img)
# print("Result from received rotated image:")
# print("Original Watermark Bits:           ", WATER_MARK_ORIG)
# print("Recovered Watermark Bits:          ", recovered_watermark)
# print("Recovered Conjugate Watermark Bits:", recovered_watermark_cong)

