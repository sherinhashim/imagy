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

def embed_watermark(image_data, watermark, N=8, T=120, delta=8):
    """
    Embed given watermark on to image.
    """
    L, K = image_data.shape
    assert L == K, "image should be square"
    pairs, cong_pairs = generate_zernike_pairs(N)
    assert len(watermark) <= len(pairs), f"Too big watermark for given N:{N}"
    WATER_MARK = watermark
    pairs = pairs[:len(WATER_MARK)]
    cong_pairs = cong_pairs[:len(WATER_MARK)]
    zern = RZern(N)
    dd = np.linspace(-1.0, 1.0, K)
    dy = np.linspace(-1.0, 1.0, L)
    xx, yy = np.meshgrid(dd, dy)

    # Create cartesian grid basis
    zern.make_cart_grid(xx, yy, unit_circle=True)
    moments, res, rnk, sv = zern.fit_cart_grid(image_data)
    z_0_0 = moments[0]
    # print("Zernike moments shape:", moments.shape)
    # print("z_0_0:", z_0_0)
    zern_moments = get_moments_for_pairs(moments, pairs, zern)
    # print("Zernike moments for selected pairs:")
    # for (n, m), val in zip(pairs, zern_moments):
    #     print(f"Moment (n={n}, m={m}): {val:.5f}")

    zern_cong_moments = get_moments_for_pairs(moments, cong_pairs, zern)
    # print("\nConjugate Zernike moments for selected pairs:")
    # for (n, m), val in zip(cong_pairs, zern_cong_moments):
    #     print(f"Moment (n={n}, m={m}): {val:.5f}")

    embedded_moments = embed_watermark_in_moments(zern_moments, z_0_0, WATER_MARK, T=T, delta=delta)
    embedded_moments_cong = embed_watermark_in_moments(zern_cong_moments, z_0_0, WATER_MARK, T=T, delta=delta)

    embedded_moments = embedded_moments - zern_moments
    embedded_moments_cong = embedded_moments_cong - zern_cong_moments

    # print("\nEmbedded Zernike moments for selected pairs:")
    # for (n, m), val in zip(pairs, embedded_moments):
    #     print(f"Moment (n={n}, m={m}): {val:.5f}")
    # print("\nEmbedded Conjugate Zernike moments for selected pairs:")
    # for (n, m), val in zip(cong_pairs, embedded_moments_cong):
    #     print(f"Moment (n={n}, m={m}): {val:.5f}")

    embeded_image_moments = np.zeros(len(moments))
    fill_moments_array(embeded_image_moments, pairs, embedded_moments, zern)
    fill_moments_array(embeded_image_moments, cong_pairs, embedded_moments_cong, zern)
    reconstructed_image = zern.eval_grid(embeded_image_moments, matrix=True)
    #print("shape of reconstruct image:", reconstructed_image.shape)
    #array_string = np.array2string(reconstructed_image, threshold=np.inf)
    #print(array_string)
    reconstructed_image[np.isnan(reconstructed_image)] = 0
    return image_data + reconstructed_image


def recover_matermark(received_image, N=8, T=120, delta=8):
    """
    recover water mark from image, based on N, T and delta
    """
    L, K = received_image.shape
    assert L == K, "image should be square"
    pairs, cong_pairs = generate_zernike_pairs(N)
    zern = RZern(N)
    dd = np.linspace(-1.0, 1.0, K)
    dy = np.linspace(-1.0, 1.0, L)
    xx, yy = np.meshgrid(dd, dy)
    zern.make_cart_grid(xx, yy, unit_circle=True)
    received_moments, _, _, _ = zern.fit_cart_grid(received_image)
    z_0_0 = received_moments[0]
    print("z_0_0:", z_0_0)
    received_zern_moments = get_moments_for_pairs(received_moments, pairs, zern)
    received_zern_cong_moments = get_moments_for_pairs(received_moments, cong_pairs, zern)
    recovered_watermark = get_water_mark_from_moments(received_zern_moments, z_0_0, T=T, delta=delta)
    recovered_watermark_cong = get_water_mark_from_moments(received_zern_cong_moments, z_0_0, T=T, delta=delta)
    return recovered_watermark, recovered_watermark_cong


