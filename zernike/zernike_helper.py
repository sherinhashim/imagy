import numpy as np
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
        results.append(moments_array[k-1])  # k is 1-based index, convert to 0-based
    return results


def embed_watermark_in_moments(moments_array, z_0_0, watermark_bits, T, delta):
    """
    Embeds watermark bits into the Zernike moments by quantization.
    """
    if(len(watermark_bits) != len(moments_array)):
        raise ValueError("Watermark bits length does not match the number of moments.")
    
    moments = np.array(moments_array)
    normalized_moments = (moments / z_0_0) * T  # Normalize by z_0_0 and scale by T
    normalized_moments_abs = np.abs(normalized_moments)
    normalized_moments_abs_frac, normalized_moments_abs_dec = np.modf(normalized_moments_abs)
    moments_q = (normalized_moments_abs_dec // delta) * delta
    moments_q = moments_q + normalized_moments_abs_frac + (0.25 * delta) + (watermark_bits * 0.5 * delta)
    modified_moments = (moments_q / normalized_moments_abs) * normalized_moments

    return modified_moments

def fill_moments_array(moments_array, pairs, modified_values, z: RZern):
    """
    Fills the modified values back into the full moments array based on the (n, m) pairs.
    """
    for (n, m), mod_val in zip(pairs, modified_values):
        k = z.nm2noll(n, m)
        moments_array[k-1] = mod_val  # Update the specific moment with the modified value

def get_water_mark_from_moments(moments_array, z_0_0, T, delta):
    """
    Extracts the watermark bits from the modified Zernike moments.
    """
    normalized_moments = (moments_array / z_0_0) * T  # Normalize by z_0_0 and scale by T
    normalized_moments_abs = np.abs(normalized_moments)
    sigma = (delta % 4) / 4
    g_r_w = np.floor(normalized_moments_abs - sigma) + sigma
    w_set = g_r_w - ((g_r_w // delta) * delta)
    check_val = delta / 2
    extracted_watermark = np.where(w_set <= check_val, 0, 1)
    return extracted_watermark