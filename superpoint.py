import cv2
import numpy as np

# ----------------------------------------------------------
# Load SuperPoint ONNX model
# ----------------------------------------------------------
onnx_model_path = r"C:\Users\Sherin\OneDrive\Desktop\mainProject\py\superpoint_lightglue_fused.onnx"
session = cv2.dnn.readNetFromONNX(onnx_model_path)

# ----------------------------------------------------------
# SuperPoint preprocessing
# ----------------------------------------------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    resized = cv2.resize(gray, (256, 256))
    blob = resized.reshape(1, 1, 256, 256)
    return blob, gray, resized

# ----------------------------------------------------------
# Run SuperPoint ONNX model
# ----------------------------------------------------------
def run_superpoint(image):

    blob, gray, resized = preprocess_image(image)

    # set input
    session.setInput(blob)

    # for your ONNX, the outputs are usually:
    # "scores" → heatmap
    # "image"  → descriptors
    outputs = session.forward(["scores", "image"])

    prob = outputs[0][0]       # (1, 65, 65) probability map
    desc = outputs[1][0]       # descriptors

    # Resize prob map back to original size
    prob = cv2.resize(prob, (gray.shape[1], gray.shape[0]))

    return prob, desc

# ----------------------------------------------------------
# Extract keypoints
# ----------------------------------------------------------
def extract_keypoints(prob, threshold=0.3):

    ys, xs = np.where(prob > threshold)
    scores = prob[ys, xs]

    pts = np.vstack((xs, ys)).T

    # sort by strength
    idx = np.argsort(scores)[::-1]
    pts = pts[idx]
    scores = scores[idx]

    return pts, scores

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    image = cv2.imread("lena.png")

    prob, desc = run_superpoint(image)

    pts, scores = extract_keypoints(prob, threshold=0.35)

    # select top 200 strongest stable points
    N = min(200, len(pts))
    stable_pts = pts[:N]

    # draw points
    out = image.copy()
    for (x, y) in stable_pts:
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imshow("Stable SuperPoint Keypoints", out)
    cv2.waitKey(0)
