from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import numpy as np
import torch

MAX_POINTS = 12

def get_radius(sd, r_tilde):
    if sd > 0.66:
        return (6/16) * r_tilde
    elif sd > 0.33:
        return (5/16) * r_tilde
    else:
        return (4/16) * r_tilde

def get_circles(image, alpha=0.8, max_point=10):
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

    inputs = processor(image, return_tensors="pt")
    W, H = image.size
    center = np.array([W/2, H/2])

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
    # ----------------------------
    # 5. SELECT NON-OVERLAPPING CIRCLES (INSIDE IMAGE)
    # ----------------------------
    selected = []

    for point, sd, desc, score in zip(sorted_pts, sorted_Sd, sorted_descriptors, sorted_scores):

        x, y = point
        radius = get_radius(sd, r_tilde)

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

        if len(selected) >= max_point:
            break

    print("Final stable non-overlapping points selected:", len(selected))
    return selected
