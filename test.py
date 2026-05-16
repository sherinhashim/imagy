from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import zernike
import helper
import matcher

DELTA     = 20       # QIM quantization step Δ
T         = 1000     # amplification factor T  (paper uses 1000)
nm_pairs   = [(1,1), (2,2), (3,1), (3,3), (4,2), (5,1), (5,3)]
watermark  = [1, 0, 1, 1, 0, 1, 0]

full_img = Image.open('s_lena.jpeg').convert("YCbCr")
img_arr = np.array(full_img)
y_img = img_arr[:,:,0]
plt.figure(figsize=(12, 5))
plt.subplot(1, 4, 1)
plt.imshow(y_img, cmap='gray')
plt.title("Original Image")

selected_points = helper.get_circles(full_img)
saved_points = []
saved_desc_list = []
saved_rad = []

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
    saved_points.append((x,y))
    saved_desc_list.append(desc)
    saved_rad.append(r)
    #print("Desc", desc) 

saved_desc = np.array(saved_desc_list)
plt.subplot(1, 4, 2)
plt.imshow(y_img, cmap='gray')
plt.title("embeded image");

img_arr[:,:,0] = y_img
new_img = Image.fromarray(img_arr, mode="YCbCr")
new_img = new_img.convert("RGB")
new_img.save("output.jpg", "JPEG")

#rotating image
print("rotated image")
rotated_arr = np.rot90(img_arr)
new_img1 = Image.fromarray(rotated_arr, mode="YCbCr")
new_img1 = new_img1.convert("RGB")
new_img1.save("rotated_output.jpg", "JPEG")

print("Image croping")
x1, y1 = 0, 0
x2, y2 = 400, 450
croped_arr = img_arr[y1:y2, x1:x2]
new_img2 = Image.fromarray(croped_arr, mode="YCbCr")
#print(croped_arr)
new_img2 = new_img2.convert("RGB")
new_img2.save("cropped_output.jpg")

y_img_rotated = rotated_arr[:,:,0]
croped_y_img = croped_arr[:,:,0]


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



#find watermark in rotated image
sorted_pts, sorted_Sd, sorted_descriptors, sorted_scores = helper.get_super_points(new_img1, 0.8)
#print("sorted_pts", sorted_pts)
matches = matcher.find_watermark_regions(saved_desc, saved_points, sorted_descriptors, sorted_pts, 'ratio')




for m in matches:
    wm_idx = m['query_idx']
    x, y   = m['keypoint']
    r = saved_rad[wm_idx]
    x1 = x-r
    x2 = x+r
    y1 = y-r
    y2 = y+r
    square = y_img_rotated[y1:y2, x1:x2]
    extracted = zernike.extract_watermark(square, nm_pairs, delta=DELTA, T=T)
    print(f"Result from received Rotated image: x:{x} y{y}, corresponds:{saved_points[wm_idx]}")
    print("Original Watermark Bits:           ", watermark)
    print("Recovered Watermark Bits:          ", extracted)

def is_circle_inside_image(x, y, r, width, height):
    return (
        x - r >= 0 and
        y - r >= 0 and
        x + r < width and
        y + r < height
    )

height, width = croped_y_img.shape[:2]    
for x, y, r, desc, score in selected_points:
    r = math.floor(r)
    if not is_circle_inside_image(x, y, r, width, height):
        continue
    print(f"Point: ({x:.1f}, {y:.1f}), Radius: {r:.1f}, Score: {score:.4f}")
    x1 = x-r
    x2 = x+r
    y1 = y-r
    y2 = y+r
    square = y_img[y1:y2, x1:x2]
    extracted = zernike.extract_watermark(square, nm_pairs, delta=DELTA, T=T)
    print("Result from croped image:")
    print("Original Watermark Bits:           ", watermark)
    print("Recovered Watermark Bits:          ", extracted)

plt.subplot(1, 4, 3)
plt.imshow(y_img_rotated, cmap='gray')
plt.title("Rotated Image")

plt.subplot(1, 4, 4)
plt.imshow(croped_y_img, cmap='gray')
plt.title("Croped Image")
plt.show()

