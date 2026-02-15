# import numpy as np
from zernike import RZern

def generate_zernike_pairs(N):
    """
    Generates (n, m) pairs for Zernike moments up to order N.
    Constraints: 
    1. n - |m| is even and |m| <= n
    2. m > 0
    3. m is not a multiple of 4 (m % 4 != 0)
    """
    pairs = []
    cong_pairs = []
    
    for n in range(N + 1):
        # Calculate all valid m for a given n
        for m in range(1, n + 1):
            # Check standard Zernike constraint: (n - m) must be even
            if (n - m) % 2 == 0:
                # Apply your specific constraint: m is not a multiple of 4
                if m % 4 != 0:
                    pairs.append((n, m))
                    cong_pairs.append((n, -m))
                    
    return pairs, cong_pairs

def get_moments_for_pairs(moments_array, pairs, z: RZern):
    """
    Retrieves the values for specific (n, m) pairs from the flattened Mahotas array.
    """
    results = []
    for n, m in pairs:
        k = z.nm2noll(n, m)
        results.append(moments_array[k])
    return results

