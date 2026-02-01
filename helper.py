image_name = "lena.png"
ALPHA = 0.8
DEG = 64
WATER_MARK = [1,0,1,0,1,0,1,1,1,1,0,0,0,0]  # Example watermark bits
T = 50  # Example threshold for Zernike moment ratios
MAX_POINTS = 12
delta = 10

import math

def get_specific_zernike(moments_array, D, target_n, target_m):
    """
    Retrieves the value for a specific (n, m) from the flattened Mahotas array.
    """
    # --- Quick Example ---
    # If moments = mahotas.features.zernike_moments(img, r, degree=9)
    # val = get_specific_zernike(moments, 9, 4, 2) 
    # print(f"Moment (4,2) is: {val}")
    # 1. Validation
    if target_n > D:
        raise ValueError(f"Target n ({target_n}) exceeds maximum degree D ({D})")
    if target_m > target_n or (target_n - target_m) % 2 != 0 or target_m < 0:
        raise ValueError(f"Invalid (n, m) pair: ({target_n}, {target_m})")

    # 2. Calculate the starting index for degree n
    # The number of non-negative m values for any degree 'i' is (i // 2) + 1
    start_idx = sum((i // 2) + 1 for i in range(target_n))
    
    # 3. Calculate the offset within that degree
    # m values increment by 2 (e.g., for n=4, m is 0, 2, 4)
    # The position of target_m is target_m // 2
    offset = target_m // 2
    
    final_idx = start_idx + offset
    return moments_array[final_idx]

def quant(value, delta):
    """
    Quantizes the given value to the nearest multiple of delta.
    """
    return math.floor(value / delta)