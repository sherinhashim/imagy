import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

# ----------------------------
# PARAMETERS
# ----------------------------
alpha = 0.8          # position > strength
MAX_POINTS = 12       # number of circles

# ----------------------------
# LOAD IMAGE
# ----------------------------
image = Image.open("lena.png").convert("RGB")
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

    if len(selected) >= MAX_POINTS:
        break

print("Final stable non-overlapping points selected:", len(selected))

# ----------------------------
# 6. DRAW CIRCLES
# ----------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(image)

for x, y, r, desc, score in selected:
    circle = plt.Circle((x, y), r, fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.scatter(x, y)
    print(f"Point: ({x:.1f}, {y:.1f}), Radius: {r:.1f}, Score: {score:.4f}, Descriptor Norm: {desc}")

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