import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('lena.png' , cv2.IMREAD_GRAYSCALE)

# Detect SIFT keypoints
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# Draw keypoints
img_kp = cv2.drawKeypoints(img, keypoints, None,
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img_kp)
plt.title('SIFT Keypoints')
plt.axis('off')
plt.show()
