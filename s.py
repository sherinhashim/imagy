"""
Complete SIFT-based embedding center selection script

Requirements:
- opencv-python (cv2) with SIFT support (OpenCV >= 4.4.0 or contrib)
- numpy
- matplotlib  (for display)

Usage:
- Set IMAGE_PATH to your image file path
- Adjust parameters (LAMBDA, ALPHA, NUM_CENTERS, etc.) as needed
- Run: python select_embedding_centers.py
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ---------- User parameters ----------
IMAGE_PATH = 'lena.png'   # <-- set your image path here
OUTPUT_VIS_PATH = 'selected_centers.png'
# Filtering & scoring
LAMBDA = 0.03        # contrast threshold (kp.response > LAMBDA)
ALPHA = 0.2          # weight for stability in S = alpha*Sm + (1-alpha)*Sd  (0..1)
# Selection params
NUM_CENTERS = 30
MIN_MARGIN = 4       # safety gap between embedding circles (pixels)
# Radius level mapping settings
RATIO_LARGE = 6.0/16.0
RATIO_MED   = 5.0/16.0
RATIO_SMALL = 4.0/16.0
# Visualization
TOP_HIGHLIGHT = 10    # top-K shown in red, others in green
SHOW_PLOT = True
# --------------------------------------

def detect_and_filter_sift(img_gray, lambda_thresh):
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img_gray, None)
    filtered = [kp for kp in kps if kp.response > lambda_thresh]
    return filtered, desc

def compute_scores(filtered_kp, image_shape):
    """
    Returns Sm, Sd, where:
    - Sm: normalized stability (response) in [0,1]
    - Sd: normalized spatial closeness to center in [0,1] (higher means closer to center)
    """
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    n = len(filtered_kp)
    if n == 0:
        return np.array([]), np.array([])

    responses = np.array([kp.response for kp in filtered_kp], dtype=np.float32)
    # normalize responses to [0,1]
    if responses.max() - responses.min() < 1e-12:
        Sm = np.ones_like(responses) * 0.5
    else:
        Sm = (responses - responses.min()) / (responses.max() - responses.min() + 1e-12)

    # distances from center: smaller dist -> higher Sd
    distances = np.array([math.hypot(kp.pt[0] - cx, kp.pt[1] - cy) for kp in filtered_kp], dtype=np.float32)
    if distances.max() - distances.min() < 1e-12:
        Sd = np.ones_like(distances) * 0.5
    else:
        Sd = 1.0 - (distances - distances.min()) / (distances.max() - distances.min() + 1e-12)

    return Sm, Sd

def assign_radius_levels(Sd_array, image_shape):
    """
    Map Sd (0..1, higher -> closer to center) to 3 radius levels based on image inscribed radius
    Returns radii array aligned with Sd_array.
    """
    h, w = image_shape[:2]
    r_tilde = min(w, h) / 2.0

    r_large = RATIO_LARGE * r_tilde
    r_med   = RATIO_MED   * r_tilde
    r_small = RATIO_SMALL * r_tilde

    radii = np.zeros_like(Sd_array, dtype=np.float32)
    # thresholds: top third -> large, middle third -> medium, bottom third -> small
    for i, sd in enumerate(Sd_array):
        if sd >= 2.0/3.0:
            radii[i] = r_large
        elif sd >= 1.0/3.0:
            radii[i] = r_med
        else:
            radii[i] = r_small
    return radii

def select_nonoverlapping_by_radii(filtered_kp, Sm, Sd, alpha,
                                   num_centers, image_shape, min_margin=4):
    """
    Combine Sm and Sd into S, assign radii based on Sd, then select top-scoring
    non-overlapping centers considering each candidate's radius.
    Returns list of dicts {'kp','score','center','radius','Sm','Sd'}.
    """
    if len(filtered_kp) == 0:
        return []

    Sm = np.asarray(Sm)
    Sd = np.asarray(Sd)
    S = alpha * Sm + (1.0 - alpha) * Sd

    radii = assign_radius_levels(Sd, image_shape)

    idx_sorted = np.argsort(-S)  # descending by score

    selected = []
    selected_centers = []  # tuples (x,y,r)

    def overlaps_with_selected(x, y, r):
        for (x2, y2, r2) in selected_centers:
            dist = math.hypot(x - x2, y - y2)
            if dist < (r + r2 + min_margin):
                return True
        return False

    for idx in idx_sorted:
        if len(selected) >= num_centers:
            break
        kp = filtered_kp[idx]
        x, y = kp.pt
        r = radii[idx]
        if not overlaps_with_selected(x, y, r):
            selected.append({
                'kp': kp,
                'score': float(S[idx]),
                'center': (float(x), float(y)),
                'radius': float(r),
                'Sm': float(Sm[idx]),
                'Sd': float(Sd[idx])
            })
            selected_centers.append((x, y, r))

    return selected

def draw_selected_centers(img_bgr, selected_centers, top_highlight=10, save_path=None, show=True):
    if len(img_bgr.shape) == 2:
        img_color = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_bgr.copy()

    for i, c in enumerate(selected_centers):
        x = int(round(c['center'][0])); y = int(round(c['center'][1])); r = int(round(c['radius']))
        color = (0,0,255) if i < top_highlight else (0,255,0)  # BGR: red top, green rest
        cv2.circle(img_color, (x,y), r, color, 2)
        cv2.circle(img_color, (x,y), 2, color, -1)
        # optional: label index (small)
        cv2.putText(img_color, str(i+1), (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, img_color)
        print(f"Saved visualization to: {os.path.abspath(save_path)}")

    if show:
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title('Selected Embedding Centers (circle = embedding radius)')
        plt.show()

    return img_color

def main():
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    img_bgr = cv2.imread(IMAGE_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) detect + lambda filter
    filtered_kp, descriptors = detect_and_filter_sift(img_gray, LAMBDA)
    print(f"Detected keypoints after λ filtering: {len(filtered_kp)}")

    if len(filtered_kp) == 0:
        print("No keypoints after filtering. Try lowering LAMBDA.")
        return

    # 2) compute Sm and Sd
    Sm, Sd = compute_scores(filtered_kp, img_gray.shape)
    print("Computed Sm and Sd.")

    # 3) select non-overlapping centers with radius levels
    selected_centers = select_nonoverlapping_by_radii(filtered_kp, Sm, Sd, ALPHA,
                                                     NUM_CENTERS, img_gray.shape, MIN_MARGIN)
    print(f"Selected {len(selected_centers)} embedding centers (requested {NUM_CENTERS}).")

    # 4) draw and save visualization
    draw_selected_centers(img_bgr, selected_centers, top_highlight=TOP_HIGHLIGHT,
                          save_path=OUTPUT_VIS_PATH, show=SHOW_PLOT)

    # 5) print selected centers info (optional)
    print("\nSelected centers (index, x, y, radius, score, Sm, Sd):")
    for i, c in enumerate(selected_centers):
        x,y = c['center']; r = c['radius']; s = c['score']
        print(f"{i+1:02d}: x={x:.1f}, y={y:.1f}, r={r:.1f}, S={s:.3f}, Sm={c['Sm']:.3f}, Sd={c['Sd']:.3f}")

if __name__ == '__main__':
    main()
