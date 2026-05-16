import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═════════════════════════════════════════════════════════════════════════════
# CORE MATCHING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def l2_normalize(descriptors):
    """
    L2-normalize a set of descriptors so each row has unit length.
    SuperPoint already outputs normalized descriptors, but safe to re-normalize
    after any processing (storage, quantization, etc.)

    descriptors : np.array of shape (N, 256)
    Returns     : np.array of shape (N, 256), each row is a unit vector
    """
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)  # (N, 1)
    norms = np.where(norms < 1e-8, 1.0, norms)                 # avoid /0
    return descriptors / norms


def cosine_similarity_matrix(desc_query, desc_db):
    """
    Compute cosine similarity between every pair of descriptors.

    Since both sets are L2-normalized, cosine_sim = dot product.
    sim[i, j] = similarity between query[i] and db[j]
    Range: [-1, 1], higher = more similar.

    desc_query : (Q, 256) — your saved watermark descriptors
    desc_db    : (N, 256) — descriptors from attacked image
    Returns    : (Q, N) float array of cosine similarities
    """
    desc_query = l2_normalize(desc_query)   # ensure unit vectors
    desc_db    = l2_normalize(desc_db)
    return desc_query @ desc_db.T           # (Q, N) dot product matrix


def match_brute_force(desc_query, desc_db, kp_db, threshold=0.7):
    """
    Method 1: Simple nearest-neighbor matching by cosine similarity.

    For each query descriptor, finds the single best matching descriptor
    in desc_db. Accepts only matches above 'threshold' similarity.

    threshold : float in [0,1]. Higher = stricter.
                SuperPoint: 0.7 is reasonable; 0.8+ is strict.

    Returns list of dicts:
        {
          'query_idx'  : index in desc_query (0..Q-1),
          'db_idx'     : index in desc_db of best match,
          'similarity' : cosine similarity score,
          'keypoint'   : (x, y) coordinate of matched point in attacked image
        }
    """
    sim_matrix = cosine_similarity_matrix(desc_query, desc_db)  # (Q, N)
    matches = []

    for q_idx in range(len(desc_query)):
        row        = sim_matrix[q_idx]          # similarities to all db points
        best_idx   = np.argmax(row)             # index of best match
        best_sim   = row[best_idx]

        if best_sim >= threshold:
            matches.append({
                'query_idx'  : q_idx,
                'db_idx'     : int(best_idx),
                'similarity' : float(best_sim),
                'keypoint'   : kp_db[best_idx]
            })

    return matches


def match_ratio_test(desc_query, desc_db, kp_db, ratio_threshold=0.8):
    """
    Method 2: Lowe's Ratio Test (best general-purpose method).

    For each query descriptor, finds the 2 best matches in desc_db.
    Accepts a match only if:
        similarity_best / similarity_second_best > ratio_threshold

    Why this works:
    - If best ≈ second_best, the match is ambiguous (two similar-looking
      regions in the attacked image) → REJECT
    - If best >> second_best, it's clearly a unique match → ACCEPT

    ratio_threshold: 0.8 (paper recommendation). Lower = stricter.

    Returns same format as match_brute_force, plus 'ratio' field.
    """
    sim_matrix  = cosine_similarity_matrix(desc_query, desc_db)  # (Q, N)
    matches = []

    for q_idx in range(len(desc_query)):
        row = sim_matrix[q_idx]

        # Get indices of top 2 matches
        if len(row) < 2:
            continue
        top2_idx  = np.argpartition(row, -2)[-2:]   # indices of 2 highest
        top2_idx  = top2_idx[np.argsort(row[top2_idx])[::-1]]  # sort desc

        best_idx  = top2_idx[0]
        sec_idx   = top2_idx[1]
        best_sim  = row[best_idx]
        sec_sim   = row[sec_idx]

        # Convert similarity to distance: dist = 1 - sim (for ratio test)
        # Ratio test in distance space: dist_best / dist_second < ratio
        # Equivalently in similarity: (1-best) / (1-second) < ratio
        dist_best = 1.0 - best_sim
        dist_sec  = 1.0 - sec_sim

        # Guard: if second distance is near zero, both are perfect matches
        if dist_sec < 1e-8:
            ratio = 1.0
        else:
            ratio = dist_best / dist_sec

        if ratio < ratio_threshold:
            matches.append({
                'query_idx'  : q_idx,
                'db_idx'     : int(best_idx),
                'similarity' : float(best_sim),
                'ratio'      : float(ratio),
                'keypoint'   : kp_db[best_idx]
            })

    return matches


def match_mutual_nearest(desc_query, desc_db, kp_db, threshold=0.6):
    """
    Method 3: Mutual Nearest Neighbor (most reliable for watermarking).

    A match (q, d) is accepted only if:
        - d is the best match for q  (forward pass)
        - q is the best match for d  (backward pass)
        - similarity >= threshold

    Why this is ideal for watermarking:
    - You have exactly 5 embedded points; you need high PRECISION.
    - False matches would cause incorrect watermark extraction.
    - Mutual check eliminates one-sided coincidental matches.

    Returns same format as other methods.
    """
    sim_matrix  = cosine_similarity_matrix(desc_query, desc_db)  # (Q, N)

    # Forward: for each query, find its best db match
    fwd_best = np.argmax(sim_matrix, axis=1)    # (Q,)  db index per query

    # Backward: for each db point, find its best query match
    bwd_best = np.argmax(sim_matrix, axis=0)    # (N,)  query index per db

    matches = []
    for q_idx in range(len(desc_query)):
        db_idx  = fwd_best[q_idx]
        sim_val = sim_matrix[q_idx, db_idx]

        # Check mutual: db_idx's best query must be q_idx
        is_mutual = (bwd_best[db_idx] == q_idx)

        if is_mutual and sim_val >= threshold:
            matches.append({
                'query_idx'  : q_idx,
                'db_idx'     : int(db_idx),
                'similarity' : float(sim_val),
                'keypoint'   : kp_db[db_idx],
                'mutual'     : True
            })

    return matches


def match_combined(desc_query, desc_db, kp_db,
                   ratio_threshold=0.8, sim_threshold=0.6):
    """
    Method 4: Ratio Test + Mutual Check combined (recommended for your case).

    Applies both filters:
        1. Lowe's ratio test (rejects ambiguous matches)
        2. Mutual nearest neighbor (rejects one-sided matches)

    This is the most conservative — you'll get fewer but highly reliable
    matches, which is what you want for watermark localization.
    """
    sim_matrix = cosine_similarity_matrix(desc_query, desc_db)

    fwd_best = np.argmax(sim_matrix, axis=1)
    bwd_best = np.argmax(sim_matrix, axis=0)

    matches = []
    for q_idx in range(len(desc_query)):
        row      = sim_matrix[q_idx]
        db_idx   = fwd_best[q_idx]
        sim_val  = row[db_idx]

        # Ratio test
        if len(row) >= 2:
            top2   = np.argpartition(row, -2)[-2:]
            top2   = top2[np.argsort(row[top2])[::-1]]
            d1     = 1.0 - row[top2[0]]
            d2     = 1.0 - row[top2[1]]
            ratio  = d1 / d2 if d2 > 1e-8 else 1.0
            if ratio >= ratio_threshold:
                continue                         # ambiguous → reject

        # Mutual check
        if bwd_best[db_idx] != q_idx:
            continue                             # not mutual → reject

        # Similarity floor
        if sim_val < sim_threshold:
            continue

        matches.append({
            'query_idx'  : q_idx,
            'db_idx'     : int(db_idx),
            'similarity' : float(sim_val),
            'keypoint'   : kp_db[db_idx]
        })

    return matches


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY: PRINT MATCH RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def print_match_results(matches, method_name, watermark_nm_pairs):
    """
    Print a table of match results, linking each matched point
    back to its watermark embedding (n,m) pair.
    """
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(f"  Matched {len(matches)} / {len(watermark_nm_pairs)} watermark points")
    print(f"  {'WM Point':>10}  {'(n,m)':>8}  {'Sim':>7}  {'Attacked kp (x,y)':>20}")
    print(f"  {'-'*55}")
    matched_idxs = {m['query_idx'] for m in matches}
    for i, nm in enumerate(watermark_nm_pairs):
        if i in matched_idxs:
            m = next(x for x in matches if x['query_idx'] == i)
            kp = m['keypoint']
            print(f"  {i:>10}  {str(nm):>8}  {m['similarity']:>7.4f}"
                  f"  ({kp[0]:6.1f}, {kp[1]:6.1f})  ✓ FOUND")
        else:
            print(f"  {i:>10}  {str(nm):>8}  {'—':>7}  {'—':>20}  ✗ not found")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def visualize_matches(original_img, attacked_img,
                      original_kps, matched_results,
                      watermark_nm_pairs, method_name="Matches"):
    """
    Draw the original watermark keypoints (left) and their matched
    locations in the attacked image (right), connected by lines.

    original_kps  : list of (x,y) — the 5 watermark embedding points
    matched_results: output of any match_* function
    """
    H = max(original_img.shape[0], attacked_img.shape[0])
    W = original_img.shape[1] + attacked_img.shape[1] + 20  # 20px gap

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if original_img.ndim == 2:
        canvas[:original_img.shape[0], :original_img.shape[1], :] = \
            np.stack([original_img]*3, axis=-1)
    else:
        canvas[:original_img.shape[0], :original_img.shape[1]] = original_img

    offset = original_img.shape[1] + 20
    if attacked_img.ndim == 2:
        canvas[:attacked_img.shape[0], offset:offset+attacked_img.shape[1]] = \
            np.stack([attacked_img]*3, axis=-1)
    else:
        canvas[:attacked_img.shape[0], offset:offset+attacked_img.shape[1]] = \
            attacked_img

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.imshow(canvas, cmap='gray' if canvas.ndim == 2 else None)

    colors = plt.cm.Set1(np.linspace(0, 1, len(original_kps)))
    matched_q = {m['query_idx']: m for m in matched_results}

    for i, (ox, oy) in enumerate(original_kps):
        color = colors[i]
        nm_label = str(watermark_nm_pairs[i])

        # Draw original keypoint (left image)
        ax.plot(ox, oy, 'o', color=color, markersize=10, markeredgewidth=2,
                markeredgecolor='white')
        ax.text(ox + 5, oy - 5, nm_label, color=color, fontsize=8,
                fontweight='bold')

        if i in matched_q:
            m       = matched_q[i]
            mx, my  = m['keypoint']
            mx_draw = mx + offset

            # Draw matched keypoint (right image)
            ax.plot(mx_draw, my, 's', color=color, markersize=10,
                    markeredgewidth=2, markeredgecolor='white')

            # Draw connecting line
            ax.plot([ox, mx_draw], [oy, my], '-', color=color,
                    linewidth=1.5, alpha=0.8)
            ax.text(mx_draw + 5, my - 5, f'{m["similarity"]:.3f}',
                    color=color, fontsize=7)
        else:
            # Unmatched: red X
            ax.plot(ox, oy, 'rx', markersize=14, markeredgewidth=2)

    ax.axvline(x=original_img.shape[1] + 10, color='white',
               linestyle='--', alpha=0.5)
    ax.set_title(f'{method_name}\n'
                 f'Left: original watermark points | '
                 f'Right: matched in attacked image',
                 fontsize=11)
    ax.axis('off')

    legend_elems = [
        mpatches.Patch(color=colors[i],
                       label=f'WM{i} {watermark_nm_pairs[i]}')
        for i in range(len(original_kps))
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'matches_{method_name.replace(" ","_")}.png', dpi=150)
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINE: How you use this in your watermarking system
# ═════════════════════════════════════════════════════════════════════════════

def find_watermark_regions(saved_descriptors, saved_keypoints,
                           attacked_descriptors, attacked_keypoints,
                           method='combined'):
    """
    Main function: given saved watermark point descriptors and the
    SuperPoint output from an attacked image, localize where the
    watermarks were embedded.

    Parameters
    ----------
    saved_descriptors  : np.array (5, 256) — descriptors saved at embed time
    saved_keypoints    : list of 5 (x,y) tuples — pixel locations at embed time
    attacked_descriptors: np.array (N, 256) — SuperPoint output on attacked image
    attacked_keypoints : list of N (x,y) tuples — SuperPoint keypoints on attacked image
    method             : 'bruteforce' | 'ratio' | 'mutual' | 'combined'

    Returns
    -------
    matches : list of match dicts (see match_* functions above)
              Each match tells you: which watermark point (query_idx)
              was found at which location (keypoint) in the attacked image.
    """
    # Ensure descriptors are float32 and normalized
    desc_q  = l2_normalize(saved_descriptors.astype(np.float32))
    desc_db = l2_normalize(attacked_descriptors.astype(np.float32))
    kp_db   = attacked_keypoints

    if method == 'bruteforce':
        return match_brute_force(desc_q, desc_db, kp_db, threshold=0.7)
    elif method == 'ratio':
        return match_ratio_test(desc_q, desc_db, kp_db, ratio_threshold=0.8)
    elif method == 'mutual':
        return match_mutual_nearest(desc_q, desc_db, kp_db, threshold=0.6)
    elif method == 'combined':
        return match_combined(desc_q, desc_db, kp_db,
                              ratio_threshold=0.8, sim_threshold=0.6)
    else:
        raise ValueError(f"Unknown method: {method}")


# ═════════════════════════════════════════════════════════════════════════════
# DEMO: Simulate the full scenario with synthetic data
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    np.random.seed(42)

    # ── Simulate saving 5 watermark descriptors at embedding time
    # In your real code: these come from SuperPoint run on original image
    N_watermark = 5
    DESC_DIM    = 256

    # 5 random unit-vector descriptors (simulating SuperPoint output)
    raw = np.random.randn(N_watermark, DESC_DIM).astype(np.float32)
    saved_descriptors = l2_normalize(raw)

    # Corresponding keypoint locations in the original image
    saved_keypoints = [
        (128, 95),
        (210, 180),
        (310, 130),
        (180, 280),
        (390, 320),
    ]

    # (n,m) pairs used for watermark embedding at each point
    watermark_nm_pairs = [(1,1), (2,2), (3,1), (3,3), (4,2)]

    # ── Simulate attacked image SuperPoint output
    # N_other random descriptors (background points in the attacked image)
    N_other = 200
    random_desc = l2_normalize(
        np.random.randn(N_other, DESC_DIM).astype(np.float32)
    )

    # Simulate the 5 watermark points surviving the attack:
    # Add small Gaussian noise to the saved descriptors (simulates
    # descriptor drift under JPEG compression / rotation / cropping)
    noise_level  = 0.05    # 0.0 = perfect match, 0.2 = heavily distorted
    survived_idx = [0, 1, 2, 4]   # simulate point 3 being cropped out

    survived_desc = []
    survived_kps  = []
    for i in survived_idx:
        noisy = saved_descriptors[i] + np.random.randn(DESC_DIM) * noise_level
        survived_desc.append(l2_normalize(noisy.reshape(1, -1))[0])
        # Keypoint location also shifts slightly under attack
        ox, oy = saved_keypoints[i]
        survived_kps.append((ox + np.random.uniform(-3, 3),
                              oy + np.random.uniform(-3, 3)))

    # Full attacked-image descriptor set = random background + survived wm points
    attacked_descriptors = np.vstack(
        [random_desc] + [d.reshape(1,-1) for d in survived_desc]
    )
    attacked_keypoints = (
        [(np.random.uniform(0,512), np.random.uniform(0,512))
         for _ in range(N_other)]
        + survived_kps
    )

    print(f"Saved watermark descriptors : {saved_descriptors.shape}")
    print(f"Attacked image descriptors  : {attacked_descriptors.shape}")
    print(f"(Points 0,1,2,4 survived attack; point 3 was cropped)")

    # ── Run all 4 methods and compare
    methods = ['bruteforce', 'ratio', 'mutual', 'combined']
    all_matches = {}

    for method in methods:
        matches = find_watermark_regions(
            saved_descriptors, saved_keypoints,
            attacked_descriptors, attacked_keypoints,
            method=method
        )
        all_matches[method] = matches
        print_match_results(matches, method.upper(), watermark_nm_pairs)

    # ── Recommendation summary
    print(f"\n{'='*60}")
    print("  RECOMMENDATION FOR YOUR WATERMARKING USE CASE")
    print(f"{'='*60}")
    print("  Use 'combined' (ratio + mutual).")
    print("  → Maximizes precision: no false watermark region detections.")
    print("  → A false match = extracting watermark from wrong location")
    print("    = wrong bits = failed copyright verification.")
    print("  → Missed match = that watermark point is skipped,")
    print("    but other surviving points still give correct bits.")
    print(f"{'='*60}")

    # ── Show similarity matrix heatmap for the 'combined' result
    sim_mat = cosine_similarity_matrix(saved_descriptors, attacked_descriptors)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(sim_mat, aspect='auto', cmap='hot', vmin=0, vmax=1)
    ax.set_xlabel("Attacked image descriptor index")
    ax.set_ylabel("Watermark point index (query)")
    ax.set_yticks(range(N_watermark))
    ax.set_yticklabels([f"WM{i} {watermark_nm_pairs[i]}" for i in range(N_watermark)])
    ax.set_title("Cosine Similarity Matrix: Watermark descriptors vs Attacked image")
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Mark the true matches with a green box
    for i, j_true in enumerate(range(N_other, N_other + len(survived_idx))):
        orig_wm_idx = survived_idx[i]
        rect = plt.Rectangle((j_true - 0.5, orig_wm_idx - 0.5),
                               1, 1, linewidth=2,
                               edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig('similarity_matrix.png', dpi=150)
    plt.show()