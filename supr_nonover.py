import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import mahotas
import helper
from math import floor
# ----------------------------
# PARAMETERS
# ----------------------------
alpha = helper.ALPHA          # position > strength

# ----------------------------
# LOAD IMAGE
# ----------------------------
image = Image.open(helper.image_name).convert("RGB")
W, H = image.size
center = np.array([W/2, H/2])

# ----------------------------
# LOAD MODEL
# ----------------------------
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# ----------------------------
# POST PROCESS
# ----------------------------
image_sizes = [(H, W)]
output = processor.post_process_keypoint_detection(outputs, image_sizes)

keypoints = output[0]["keypoints"].detach().cpu().numpy()  # (N,2)
scores = output[0]["scores"].detach().cpu().numpy()        # (N,)
descriptors = output[0]["descriptors"].detach().cpu().numpy()  # (N,256)

print("Total keypoints detected:", len(keypoints))

# ----------------------------
# 1. STABILITY SCORE (Sm)
# ----------------------------
Sm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# ----------------------------
# 2. DISTANCE SCORE (Sd)
# ----------------------------
distances = np.linalg.norm(keypoints - center, axis=1)
Sd = 1 - (distances / (distances.max() + 1e-8))

# ----------------------------
# 3. COMPREHENSIVE SCORE
# ----------------------------
S = (1 - alpha) * Sm + alpha * Sd

# sort strongest
sorted_idx = np.argsort(-S)
sorted_pts = keypoints[sorted_idx]
sorted_Sd = Sd[sorted_idx]
sorted_descriptors = descriptors[sorted_idx]
sorted_scores = S[sorted_idx]

# ----------------------------
# 4. INSCRIBED IMAGE RADIUS (r~)
# ----------------------------
r_tilde = min(W, H) / 2

def get_radius(sd):
    if sd > 0.66:
        return (6/16) * r_tilde
    elif sd > 0.33:
        return (5/16) * r_tilde
    else:
        return (4/16) * r_tilde

# ----------------------------
# 5. SELECT NON-OVERLAPPING CIRCLES (INSIDE IMAGE)
# ----------------------------
selected = []

for point, sd, desc, score in zip(sorted_pts, sorted_Sd, sorted_descriptors, sorted_scores):

    x, y = point
    radius = get_radius(sd)

    # Must be INSIDE the image
    if (x - radius < 0) or (x + radius > W) or (y - radius < 0) or (y + radius > H):
        continue

    overlap = False
    for (px, py, pr, pd, ps) in selected:
        dist = np.linalg.norm([x - px, y - py])
        if dist < (radius + pr):
            overlap = True
            break

    if not overlap:
        selected.append((x, y, radius, desc, score))

    if len(selected) >= helper.MAX_POINTS:
        break

print("Final stable non-overlapping points selected:", len(selected))

# ----------------------------
# 6. DRAW CIRCLES
# ----------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(image)
greyImage = np.array(image.convert("L"))
#zerikes = []
waterMark = helper.WATER_MARK
waterMarkLen = len(waterMark)

for x, y, r, desc, score in selected:
    circle = plt.Circle((x, y), r, fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.scatter(x, y)
    #print(f"Point: ({x:.1f}, {y:.1f}), Radius: {r:.1f}, Score: {score:.4f}, Descriptor Norm: {desc}")
    print(f"Point: ({x:.1f}, {y:.1f}), Radius: {r:.1f}, Score: {score:.4f}")
    value = mahotas.features.zernike_moments(greyImage, r, degree=helper.DEG, cm=(y, x))
    #print(f"Zernike Moments (degree 64) at this point: {value}")
    #zerikes.append(value)
    c_n_m = []
    for n in range(0, helper.DEG, 1):
        for m in range(0, n+1, 1):
            if (n - m) % 2 != 0 or m < 0:
                continue
            if m % 4 != 0:
                val = helper.get_specific_zernike(value, helper.DEG, n, m)
                c_n_m.append((n, m, val))
                #print(f"  Moment ({n},{m}): {val:.4f}")
    #TODO can modify the key here to select different moments
    CK = c_n_m[0:waterMarkLen]
    # print("  Watermark bits and corresponding moments:")
    # for i, (n, m, val) in enumerate(CK):
    #     bit = waterMark[i]
    #     print(f"    Bit: {bit}, Moment ({n},{m}): {val:.4f}")
    z_0_0 = helper.get_specific_zernike(value, helper.DEG, 0, 0)
    # print(f"  z_0 (0,0) Moment: {z_0_0:.4f}")
    z_r = [((val/z_0_0)*helper.T, n, m) for (n, m, val) in CK]
    # print("  Scaled Zernike moments for watermark embedding:")
    # for scaled_val, n, m in z_r:
    #     print(f"    Moment ({n},{m}): {scaled_val:.4f}")
    z_w = []
    for i, (z_val, n, m) in enumerate(z_r):
        bit = waterMark[i]
        q_val = helper.quant(floor(abs(z_val)), helper.delta) * helper.delta
        D = abs(z_val) - floor(abs(z_val))
        val = 0
        if bit == 1:
            val = q_val + (3*(helper.delta / 4)) + D
        else:
            val = q_val + (helper.delta / 4) + D
        val = (val / abs(z_val)) * z_val
        z_w.append((val, n, m))
        print(f"    Modified Moment ({n},{m}) for Bit {bit}: {val:.4f}")
        # print(f"      Bit: {bit}, Quantized Value: {q_val}")

plt.axis("off")
plt.savefig("non_overlapping_circles_old.png", dpi=300)
plt.close()

# ----------------------------
# 7. CREATE NON-CIRCULAR (COMPENSATION) MASK
# ----------------------------
mask = np.ones((H, W), dtype=np.uint8) * 255   # white = compensation area

for x, y, r, desc, score in selected:
    for i in range(H):
        for j in range(W):
            if (j - x) ** 2 + (i - y) ** 2 <= r ** 2:
                mask[i, j] = 0   # black = circle zone

plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.savefig("compensation_mask_old.png", dpi=300)
plt.close()

print("Saved: non_overlapping_circles_old.png")
print("Saved: compensation_mask_old.png")