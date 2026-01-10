import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

# -------- Load model --------
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
model.eval()

# -------- Load image --------
img = cv2.imread("test.jpg")   # <-- change name if needed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------- Preprocess --------
inputs = processor(images=gray, return_tensors="pt")

# -------- Get output --------
with torch.no_grad():
    outputs = model(**inputs)

# -------- Extract keypoints and descriptors --------
keypoints = outputs.keypoints[0].cpu().numpy()     # (N,2)
descriptors = outputs.descriptors[0].cpu().numpy() # (256,N)

print("Total keypoints detected:", keypoints.shape[0])

# --------- Select STABLE POINTS ---------
# Compute strength of descriptors
strength = np.linalg.norm(descriptors, axis=0)

# Top 100 stable points
top_k = 100
stable_idx = np.argsort(strength)[-top_k:]

stable_points = keypoints[stable_idx]

print("Stable points selected:", stable_points.shape[0])

# -------- Plot points --------
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Plot all keypoints (blue)
plt.scatter(keypoints[:,0], keypoints[:,1], s=5, color="blue")

# Plot stable keypoints (red)
plt.scatter(stable_points[:,0], stable_points[:,1], s=20, color="red")

plt.title("SuperPoint Keypoints (Blue) and Stable Points (Red)")
plt.axis("off")
plt.show()
