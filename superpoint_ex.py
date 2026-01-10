from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import matplotlib.pyplot as plt
import torch
from PIL import Image
#import requests

imp_path = "lena.png"
image = Image.open(imp_path)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)

image_sizes = image.size[::-1]  # (height, width)
print("Image size (H, W):", image_sizes)
output = processor.post_process_keypoint_detection(outputs, [image_sizes])
#print("Number of keypoints detected:", len(output[0]["keypoints"]))
#print("Keypoints:", output[0]["keypoints"])
#print("Scores:", output[0]["scores"])
#print("Descriptors shape:", output[0]["descriptors"].shape)
#print(outputs["keypoints"][0])
keypoints = output[0]["keypoints"]
scores = output[0]["scores"]
descriptors = output[0]["descriptors"]
# print(keypoints)
# print(scores)
# print(descriptors)  
x_tensor = keypoints[:, 0]
y_tensor = keypoints[:, 1]


plt.axis("off")
plt.imshow(image)
plt.scatter(
    keypoints[:, 0].detach().numpy(),
    keypoints[:, 1].detach().numpy(),
    c=scores.detach().numpy() * 100,
    s=scores.detach().numpy() * 50,
    alpha=0.8
)
plt.savefig(f"output_image.png")
