# pip install numpy scikit-image matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Zernike Basis and Moments (from scratch, no library needed)
# ═════════════════════════════════════════════════════════════════════════════

def zernike_radial(n, m, rho):
    """
    Radial polynomial R_{n,|m|}(rho)  [Paper Eq. 3]
    n   : order (int >= 0)
    m   : repetition (int); uses |m| internally
    rho : 2D array of radial distances in [0, 1]
    Returns real 2D array.
    """
    m_abs = abs(m)
    R = np.zeros_like(rho, dtype=float)
    for c in range((n - m_abs) // 2 + 1):
        num = (-1)**c * factorial(n - c)
        den = (factorial(c) *
               factorial((n + m_abs) // 2 - c) *
               factorial((n - m_abs) // 2 - c))
        R += (num / den) * rho ** (n - 2 * c)
    return R


def zernike_basis(n, m, size):
    """
    Complex Zernike basis V_{n,m}(x,y)  [Paper Eq. 2]
    V_{n,m} = R_{n,m}(rho) * exp(j*m*theta)
    Pixels outside the unit circle are zero.
    Returns complex 2D array of shape (size, size).
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y   = np.meshgrid(x, y)
    rho    = np.sqrt(X**2 + Y**2)
    theta  = np.arctan2(Y, X)
    mask   = (rho <= 1.0)

    R = zernike_radial(n, m, rho)
    V = np.zeros((size, size), dtype=complex)
    V[mask] = R[mask] * np.exp(1j * m * theta[mask])
    return V


def zernike_moment(image, n, m, size=None):
    """
    Compute Z_{n,m} of image using discrete approximation  [Paper Eq. 5]
    Z_{n,m} ≈ (n+1)/π * Σ V*_{n,m}(xi,yi) * f(xi,yi) * Δx * Δy
    Returns complex scalar.
    """
    if size is None:
        size = image.shape[0]
    V_conj = np.conj(zernike_basis(n, m, size))
    dx     = 2.0 / size                 # Δx = Δy = 2/N
    Znm    = ((n + 1) / np.pi) * np.sum(V_conj * image * dx * dx)
    return Znm


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Z00 Normalization  [Paper Eq. 14]
# ═════════════════════════════════════════════════════════════════════════════

def normalize_moment(Znm, Z00, T=1000):
    """
    Normalize Z_{n,m} by Z_{0,0} for scale invariance  [Paper Eq. 14]

        Z^R_{n,m} = (Z_{n,m} / Z_{0,0}) * T

    Z00 : zero-order moment (real scalar, acts as image energy)
    T   : amplification factor (paper uses T=1000)
          Brings the small ratio into a range where integer QIM works.
    Returns complex scalar Z^R_{n,m}
    """
    if abs(Z00) < 1e-12:
        raise ValueError("Z00 is near zero — image patch is empty or all zeros.")
    return (Znm / Z00) * T


def denormalize_moment(ZR_nm, Z00, T=1000):
    """
    Reverse of normalize_moment.
    Recover Z_{n,m} from Z^R_{n,m}:
        Z_{n,m} = Z^R_{n,m} * Z_{0,0} / T
    Used during image reconstruction.
    """
    return ZR_nm * Z00 / T


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QIM Watermark Embedding  [Paper Eq. 15, 17]
# ═════════════════════════════════════════════════════════════════════════════

def embed_bit_qim(ZR_nm, bit, delta=20):
    """
    Embed one watermark bit into |Z^R_{n,m}|  [Paper Eq. 15]

    Operates on the NORMALIZED moment Z^R_{n,m} (after Eq. 14).
    Only the magnitude is modified; phase is preserved.

    QIM rule:
        bit=1 → new_mag = ⌊|Z^R|/Δ⌋ * Δ + (3/4)*Δ + Ddec
        bit=0 → new_mag = ⌊|Z^R|/Δ⌋ * Δ + (1/4)*Δ + Ddec
    where Ddec = fractional part of |Z^R|

    Returns Z^Rw_{n,m}  (watermarked normalized moment)
    """
    mag  = abs(ZR_nm)
    Ddec = mag - np.floor(mag)          # fractional part kept unchanged

    floor_div = np.floor(mag / delta)

    if bit == 1:
        new_mag = floor_div * delta + (3/4) * delta + Ddec
    else:
        new_mag = floor_div * delta + (1/4) * delta + Ddec

    # Scale to new magnitude, keep original phase  [Eq. 17]
    if mag < 1e-12:
        return complex(new_mag, 0)
    return (new_mag / mag) * ZR_nm


def get_quantization_error(ZR_nm, delta=20):
    """
    Compute quantization error eq_i  [Paper Eq. 20]

    eq_i = ⌊|Z^R_{n,m}|⌋ - Q(⌊|Z^R_{n,m}|⌋, Δ)
    where Q(x, Δ) = ⌊x/Δ⌋  (the quantization operation)

    This is the integer-part error introduced by QIM.
    Needed for reversible recovery of the cover image.
    """
    mag        = abs(ZR_nm)
    floor_mag  = np.floor(mag)
    q_val      = np.floor(floor_mag / delta) * delta   # Q(⌊|Z^R|⌋, Δ)
    eq         = floor_mag - q_val
    return eq


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Watermark Bit Extraction  [Paper Eq. 27, 28]
# ═════════════════════════════════════════════════════════════════════════════

def extract_bit_qim(ZRw_nm, delta=20):
    """
    Extract watermark bit from a (possibly attacked) normalized moment.

    Step 1 — Remove the effect of Ddec (fractional part)  [Paper Eq. 27]:
        sigma    = (Δ mod 4) / 4
        G^Rw     = ⌊|Z^Rw| - sigma⌋ + sigma

        This isolates the integer-part quantization pattern,
        removing the floating Ddec that was preserved during embedding.

    Step 2 — Decision rule  [Paper Eq. 28]:
        residue = G^Rw - Q(G^Rw, Δ) * Δ    (i.e., G^Rw mod Δ)
        if residue <= Δ/2  →  bit = 0
        if residue >  Δ/2  →  bit = 1

    Returns extracted bit (0 or 1)
    """
    mag   = abs(ZRw_nm)

    # ── Eq. 27: remove Ddec effect
    sigma = (delta % 4) / 4             # for delta divisible by 4, sigma=0
    G     = np.floor(mag - sigma) + sigma

    # ── Eq. 28: decision
    residue = G - np.floor(G / delta) * delta   # G mod Δ
    return 1 if residue > delta / 2 else 0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — I_rw Computation  [Paper Eq. 18, 19]
# ═════════════════════════════════════════════════════════════════════════════

def compute_Irw(image, nm_pairs, watermark_bits, delta=20, T=1000):
    """
    Compute the watermark-change image I_rw  [Paper Eq. 18]
    and the watermarked image Iw              [Paper Eq. 19]

    Full pipeline for one local circular region:
        1. Compute Z_{0,0}  (used for normalization, Eq. 14)
        2. For each (n,m):
            a. Compute Z_{n,m}  and  Z_{n,-m}          [Eq. 5]
            b. Normalize: Z^R = Z / Z00 * T             [Eq. 14]
            c. Embed bit:  Z^Rw via QIM                 [Eq. 15, 17]
            d. Denormalize back: Z^w = Z^Rw * Z00 / T  [inverse of Eq. 14]
            e. Accumulate: (Z^w - Z) * V                [Eq. 18]
        3. I_rw = sum of all (Z^w - Z) * V
        4. I_w  = I + I_rw                              [Eq. 19]

    Returns
    -------
    Irw      : 2D real array — pixel-domain change
    Iw       : 2D real array — watermarked patch
    cache    : dict with all intermediate values for extraction/recovery
    """
    size = image.shape[0]
    Irw  = np.zeros((size, size), dtype=complex)

    # Step 1: compute Z_{0,0} once — used for ALL moment normalizations
    Z00 = zernike_moment(image, 0, 0, size)    # real-valued, image energy
    print(f"  Z_00 = {Z00.real:.4f}  (image energy in unit circle)")

    cache = {'Z00': Z00, 'moments': {}}

    for (n, m), bit in zip(nm_pairs, watermark_bits):

        # ── Step 2a: compute original moments for (n,m) and (n,-m)
        Znm  = zernike_moment(image, n,  m, size)
        Znm_ = zernike_moment(image, n, -m, size)

        # ── Step 2b: normalize by Z00  [Eq. 14]
        ZR_nm  = normalize_moment(Znm,  Z00, T)
        ZR_nm_ = normalize_moment(Znm_, Z00, T)

        # ── Step 2c: embed bit via QIM on normalized moment  [Eq. 15]
        ZRw_nm  = embed_bit_qim(ZR_nm,  bit, delta)
        ZRw_nm_ = embed_bit_qim(ZR_nm_, bit, delta)

        # ── compute quantization errors (needed for reversibility, Eq. 20)
        eq  = get_quantization_error(ZR_nm,  delta)
        eq_ = get_quantization_error(ZR_nm_, delta)

        # ── Step 2d: denormalize back to original moment scale
        Znm_w  = denormalize_moment(ZRw_nm,  Z00, T)
        Znm_w_ = denormalize_moment(ZRw_nm_, Z00, T)

        # ── Step 2e: get basis functions and accumulate Eq. 18
        Vnm  = zernike_basis(n,  m, size)
        Vnm_ = zernike_basis(n, -m, size)

        Irw += (Znm_w - Znm) * Vnm + (Znm_w_ - Znm_) * Vnm_

        # save everything for later extraction / analysis
        cache['moments'][(n, m)] = {
            'Znm':    Znm,   'ZR_nm':  ZR_nm,   'ZRw_nm': ZRw_nm,
            'Znm_w':  Znm_w, 'eq':     eq,       'bit':    bit,
        }
        cache['moments'][(n, -m)] = {
            'Znm':    Znm_,  'ZR_nm':  ZR_nm_,  'ZRw_nm': ZRw_nm_,
            'Znm_w':  Znm_w_,'eq':     eq_,      'bit':    bit,
        }

    Irw_real = np.real(Irw)
    Iw       = image + Irw_real         # Eq. 19

    return Irw_real, Iw, cache


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Watermark Extraction  [Paper Eq. 27, 28]
# ═════════════════════════════════════════════════════════════════════════════

def extract_watermark(Iw, nm_pairs, delta=20, T=1000):
    """
    Extract watermark bits from the watermarked image Iw.

    Pipeline:
        1. Recompute Z00 from Iw (slightly changed from original)
        2. For each (n,m): compute Z_{n,m} from Iw, normalize, apply Eq. 27+28

    Returns list of extracted bits.
    """
    size = Iw.shape[0]

    # Z00 from the watermarked image (slightly different from original)
    Z00_w = zernike_moment(Iw, 0, 0, size)

    extracted_bits = []
    for (n, m) in nm_pairs:
        Znm_w  = zernike_moment(Iw, n, m, size)
        ZRw_nm = normalize_moment(Znm_w, Z00_w, T)   # normalize [Eq. 14]
        bit    = extract_bit_qim(ZRw_nm, delta)       # decision  [Eq. 27, 28]
        extracted_bits.append(bit)

    return extracted_bits


# # ═════════════════════════════════════════════════════════════════════════════
# # SECTION 7 — MAIN: Run the Full Pipeline
# # ═════════════════════════════════════════════════════════════════════════════

# # ── Parameters (matching paper Table I)
# # SIZE      = 64       # local circular region size (pixels)
# DELTA     = 20       # QIM quantization step Δ
# T         = 1000     # amplification factor T  (paper uses 1000)

# # # ── Load image patch
# # img_full = data.camera().astype(float)
# # cx, cy   = 256, 256
# # h        = SIZE // 2
# # patch    = img_full[cx-h:cx+h, cy-h:cy+h].copy()

# # Load a grayscale image and extract a square patch (local circular region)
# from PIL import Image
# full_img = Image.open('s_lena.jpeg').convert("YCbCr")
# img_arr = np.array(full_img)
# y_img = img_arr[:,:,0]
# patch    = y_img


# # ── (n,m) pairs: constraints from paper:
# #    0 < m <= n,  (n - |m|) is even,  m ≠ 4j (j integer)
# nm_pairs   = [(1,1), (2,2), (3,1), (3,3), (4,2), (5,1), (5,3)]
# watermark  = [1, 0, 1, 1, 0, 1, 0]

# print("=" * 55)
# print("EMBEDDING")
# print("=" * 55)
# Irw, Iw, cache = compute_Irw(patch, nm_pairs, watermark, delta=DELTA, T=T)

# mse  = np.mean((patch - Iw)**2)
# psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
# print(f"\nMax |I_rw| pixel change : {np.max(np.abs(Irw)):.4f}")
# print(f"PSNR of watermarked img : {psnr:.2f} dB")

# # ── Print moment table
# print(f"\n{'(n,m)':<8} {'|Z_nm|':>10} {'|Z^R_nm|':>10} "
#       f"{'|Z^Rw_nm|':>11} {'eq':>6} {'bit':>5}")
# print("-" * 55)
# for (n, m) in nm_pairs:
#     d = cache['moments'][(n, m)]
#     print(f"({n},{m:<2})   {abs(d['Znm']):>10.2f} {abs(d['ZR_nm']):>10.4f} "
#           f"{abs(d['ZRw_nm']):>11.4f} {d['eq']:>6.1f} {d['bit']:>5}")

# print("=" * 55)
# print("EXTRACTION  [Eq. 27, 28]")
# print("=" * 55)
# extracted = extract_watermark(Iw, nm_pairs, delta=DELTA, T=T)
# ber = sum(a != b for a, b in zip(watermark, extracted)) / len(watermark)
# print(f"Original  : {watermark}")
# print(f"Extracted : {extracted}")
# print(f"BER       : {ber:.3f}  ({'PERFECT' if ber == 0 else 'ERRORS FOUND'})")

# # ── Verification: extract bit-by-bit with detail
# print(f"\n{'(n,m)':<8} {'|ZRw| from Iw':>15} {'sigma':>7} "
#       f"{'G^Rw':>10} {'residue':>9} {'bit':>5}")
# print("-" * 60)
# size_  = Iw.shape[0]
# Z00_w  = zernike_moment(Iw, 0, 0, size_)
# for (n, m) in nm_pairs:
#     Znm_w   = zernike_moment(Iw, n, m, size_)
#     ZRw     = normalize_moment(Znm_w, Z00_w, T)
#     mag     = abs(ZRw)
#     sigma   = (DELTA % 4) / 4
#     G       = np.floor(mag - sigma) + sigma
#     residue = G - np.floor(G / DELTA) * DELTA
#     bit     = 1 if residue > DELTA / 2 else 0
#     print(f"({n},{m:<2})   {mag:>15.4f} {sigma:>7.3f} {G:>10.4f} "
#           f"{residue:>9.4f} {bit:>5}")

# # ═════════════════════════════════════════════════════════════════════════════
# # SECTION 8 — Visualization
# # ═════════════════════════════════════════════════════════════════════════════

# fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# axes[0].imshow(patch,             cmap='gray', vmin=0, vmax=255)
# axes[0].set_title('Original I'); axes[0].axis('off')

# axes[1].imshow(Irw,               cmap='RdBu')
# axes[1].set_title('I_rw  [Eq.18]\n(watermark change)'); axes[1].axis('off')

# axes[2].imshow(np.clip(Iw,0,255), cmap='gray', vmin=0, vmax=255)
# axes[2].set_title(f'Watermarked Iw\n[Eq.19]  PSNR={psnr:.1f}dB')
# axes[2].axis('off')

# axes[3].imshow(np.abs(patch - Iw)*10, cmap='hot')
# axes[3].set_title('|I − Iw| × 10\n(amplified diff)'); axes[3].axis('off')

# plt.tight_layout()
# plt.savefig('watermark_result.png', dpi=150)
# plt.show()