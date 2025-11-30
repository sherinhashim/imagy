import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
cx, cy = w / 2, h / 2  # image center

# Step 1: Detect SIFT keypoints
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# Step 2: Apply contrast threshold (λ)
lambda_threshold = 0.03
filtered_kp = [kp for kp in keypoints if kp.response > lambda_threshold]

# Step 3: Compute Stability Strength (Sm)
responses = np.array([kp.response for kp in filtered_kp])
Sm = (responses - responses.min()) / (responses.max() - responses.min() + 1e-8)

# Step 4: Compute Spatial Score (Sd)
distances = np.array([np.sqrt((kp.pt[0] - cx)**2 + (kp.pt[1] - cy)**2) for kp in filtered_kp])
Sd = 1 - (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
# ↑ closer to center → higher Sd

# Step 5: Combine both scores into S
alpha = 0.7  # weight for stability
beta = 0.3   # weight for position
S = alpha * Sm + beta * Sd

# Step 6: Sort by combined score
sorted_indices = np.argsort(-S)
final_kp = [filtered_kp[i] for i in sorted_indices[:50]]  # top 50 points

# Step 7: Visualize
img_final = cv2.drawKeypoints(img, final_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(8,8))
plt.imshow(img_final, cmap='gray')
plt.title("Top Robust & Well-Located Keypoints")
plt.axis('off')
plt.show()
