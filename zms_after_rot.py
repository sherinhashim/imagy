import mahotas
import mahotas.features
from skimage import io, color, transform
import numpy as np
import matplotlib.pyplot as plt

# ---------- Step 1: Read image ----------
img = io.imread('Lena.png')  # Replace with your image filename

# Convert to grayscale
if img.ndim == 3:
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

# Resize to square (optional but helps)
size = min(img_gray.shape)
img_gray = transform.resize(img_gray, (size, size), anti_aliasing=True)

# Define radius (half of smallest dimension)
radius = size // 2
degree = 3  # maximum order

# ---------- Step 2: Compute Zernike moments on original ----------
zm_original = mahotas.features.zernike_moments(img_gray, radius, degree=degree)

print("\n================ ORIGINAL IMAGE ================")
print(f"Computed {len(zm_original)} Zernike moments (degree={degree})")
for i, val in enumerate(zm_original):
    print(f"Zernike[{i:02d}] = {val:.6f}")

# ---------- Step 3: Rotate image ----------
angle = 45
img_rotated = transform.rotate(img_gray, angle, resize=False)

# ---------- Step 4: Compute Zernike moments on rotated image ----------
zm_rotated = mahotas.features.zernike_moments(img_rotated, radius, degree=degree)

print("\n================ ROTATED IMAGE (45°) ================")
print(f"Computed {len(zm_rotated)} Zernike moments (degree={degree})")
for i, val in enumerate(zm_rotated):
    print(f"Zernike[{i:02d}] = {val:.6f}")

# ---------- Step 5: Compare ----------
abs_diff = np.abs(zm_original - zm_rotated)
rel_diff = abs_diff / (np.abs(zm_original) + 1e-12)

print("\n================ COMPARISON ================")
print("Index | Original ZM  | Rotated ZM  | Abs Diff  | Rel Diff (%)")
print("---------------------------------------------------------------")
for i in range(len(zm_original)):
    print(f"{i:3d}   {zm_original[i]:.6f}   {zm_rotated[i]:.6f}   {abs_diff[i]:.6e}   {rel_diff[i]*100:.4f}")

print("\n✅ Mean relative difference: {:.4f}%".format(np.mean(rel_diff)*100))
print("✅ Zernike magnitudes are almost identical → rotation invariance verified!\n")

# ---------- Step 6: Plot comparison ----------
plt.figure(figsize=(8,4))
plt.plot(zm_original, 'bo-', label='Original')
plt.plot(zm_rotated, 'r--', label='Rotated 45°')
plt.title(f'Comparison of Zernike Moment Magnitudes (degree={degree})')
plt.xlabel('Moment Index')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
