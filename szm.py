import cv2
import numpy as np
import mahotas
import matplotlib.pyplot as plt
from math import sqrt

# ============================================================
# 1️⃣ Detect keypoints using Harris or Shi-Tomasi
# ============================================================
def detect_keypoints(image_gray):
    # Use Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(image_gray, maxCorners=500, qualityLevel=0.01, minDistance=5)
    if corners is None:
        return []
    keypoints = [cv2.KeyPoint(float(x), float(y), 3) for [[x, y]] in corners]
    print(f"Detected {len(keypoints)} initial keypoints.")
    return keypoints

# ============================================================
# 2️⃣ Compute Sm (mean intensity) and Sd (standard deviation)
# ============================================================
def compute_Sm_Sd(image_gray, keypoints, window_size=7):
    Sm, Sd = [], []
    pad = window_size // 2
    padded = cv2.copyMakeBorder(image_gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = padded[y:y+window_size, x:x+window_size]
        Sm.append(np.mean(patch))
        Sd.append(np.std(patch))
    print("Computed Sm and Sd.")
    return np.array(Sm), np.array(Sd)

# ============================================================
# 3️⃣ λ-based filtering (feature strength)
# ============================================================
def filter_keypoints_lambda(keypoints, Sm, Sd, lamda_thresh=0.5):
    lamda = Sd / (Sm + 1e-6)
    mean_lamda = np.mean(lamda)
    filtered = [kp for kp, l in zip(keypoints, lamda) if l > lamda_thresh * mean_lamda]
    print(f"Detected keypoints after λ filtering: {len(filtered)}")
    return filtered

# ============================================================
# 4️⃣ Select embedding centers
# ============================================================
def select_embedding_centers(filtered_keypoints, num_centers=30, min_distance=30):
    centers = []
    for kp in filtered_keypoints:
        x, y = kp.pt
        if all(sqrt((x - c['center'][0])**2 + (y - c['center'][1])**2) > min_distance for c in centers):
            centers.append({'center': (x, y), 'radius': 20 + np.random.randint(5, 15)})
        if len(centers) >= num_centers:
            break
    print(f"Selected {len(centers)} embedding centers (requested {num_centers}).")
    return centers

# ============================================================
# 5️⃣ Visualize selected centers
# ============================================================
def visualize_selected_centers(image, centers, save_path="selected_centers.png"):
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in centers:
        cv2.circle(vis_img, (int(c['center'][0]), int(c['center'][1])), int(c['radius']), (0, 255, 0), 1)
        cv2.circle(vis_img, (int(c['center'][0]), int(c['center'][1])), 2, (0, 0, 255), -1)
    cv2.imwrite(save_path, vis_img)
    print(f"Saved visualization to: {save_path}")

# ============================================================
# 6️⃣ Compute Zernike Moments (robust)
# ============================================================
def compute_zernike_for_regions(img_gray, selected_centers, order=8):
    h, w = img_gray.shape
    results = []

    for i, c in enumerate(selected_centers):
        x, y = int(round(c['center'][0])), int(round(c['center'][1]))
        r = int(round(c['radius']))

        if r < 5:
            print(f"[{i+1:02d}] Skipping tiny region at ({x},{y}) r={r}")
            continue

        x1, x2 = max(0, x - r), min(w, x + r)
        y1, y2 = max(0, y - r), min(h, y + r)
        if x2 <= x1 or y2 <= y1:
            print(f"[{i+1:02d}] Invalid patch bounds for center ({x},{y}), r={r}")
            continue

        patch = img_gray[y1:y2, x1:x2].astype(np.float32)
        if patch.max() - patch.min() < 1e-6:
            patch_norm = np.zeros_like(patch)
        else:
            patch_norm = (patch - patch.min()) / (patch.max() - patch.min())

        h_p, w_p = patch_norm.shape
        yy, xx = np.ogrid[:h_p, :w_p]
        cy, cx = h_p / 2.0, w_p / 2.0
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2

        if np.sum(mask) < 10:
            print(f"[{i+1:02d}] Too few pixels in mask at ({x},{y}), skipped.")
            continue

        patch_masked = np.zeros_like(patch_norm)
        patch_masked[mask] = patch_norm[mask]
        radius = min(h_p, w_p) / 2.0

        try:
            zm = mahotas.features.zernike_moments(patch_masked, radius, degree=order)
            results.append({
                'center': (x, y),
                'radius': r,
                'zernike_moments': zm
            })
            print(f"[{i+1:02d}] Computed {len(zm)} Zernike moments at ({x},{y}), r={r}")
        except Exception as e:
            print(f"[{i+1:02d}] Error computing ZMs at ({x},{y}): {e}")

    return results

# ============================================================
# 7️⃣ Main Execution
# ============================================================
if __name__ == "__main__":
    image_path = "test_image.png"   # 🔸 change to your image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1–3: Detect and filter keypoints
    kps = detect_keypoints(image)
    Sm, Sd = compute_Sm_Sd(image, kps)
    filtered_kps = filter_keypoints_lambda(kps, Sm, Sd, lamda_thresh=0.5)

    # Step 4–5: Select centers and visualize
    centers = select_embedding_centers(filtered_kps, num_centers=30, min_distance=40)
    visualize_selected_centers(image, centers, "selected_centers.png")

    # Step 6: Compute Zernike moments
    zm_results = compute_zernike_for_regions(image, centers, order=8)

    # Step 7: Print summary
    print("\n=== Zernike Moments Summary ===")
    for r in zm_results:
        c = r['center']
        z = r['zernike_moments']
        print(f"Center {c} (r={r['radius']}) → First 5 ZMs: {np.round(z[:5], 4)}")
