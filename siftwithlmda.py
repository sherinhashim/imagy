import cv2
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Detect SIFT keypoints
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# Set contrast threshold λ
lambda_threshold = 0.04 # adjust this value

# Filter keypoints based on response (contrast)
filtered_kp = [kp for kp in keypoints if kp.response > lambda_threshold]

print(f"Original keypoints: {len(keypoints)}")
print(f"Filtered keypoints: {len(filtered_kp)}")
img_with_kp = cv2.drawKeypoints(
    img, 
    filtered_kp, 
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Step 7: Display using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(img_with_kp, cmap='gray')
plt.title(f"SIFT Keypoints after filtering (λ = {lambda_threshold})")
plt.axis('off')
plt.show()
