"""Image array manipulation."""

import numpy as np

def create_image(seed=42):
    """Create a synthetic image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

def manipulate(img):
    print(f"Image shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Min: {img.min()}, Max: {img.max()}")
    print(f"Memory: {img.nbytes / 1024:.1f} KB")
    
    # Convert to grayscale
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = img @ weights
    print(f"\nGrayscale shape: {gray.shape}")
    print(f"Gray range: {gray.min():.2f} to {gray.max():.2f}")
    
    # Brightness adjustment
    brightened = np.clip(img.astype(int) + 50, 0, 255).astype(np.uint8)
    print(f"\nBrightened image shape: {brightened.shape}")
    
    # Contrast
    contrast = np.clip(img.astype(float) * 1.5, 0, 255).astype(np.uint8)
    
    # Flip (horizontal)
    flipped = img[:, ::-1]
    
    # Rotate 90 degrees
    rotated = np.rot90(img)
    print(f"Rotated shape: {rotated.shape}")
    
    # Crop center
    h, w = img.shape[:2]
    ch, cw = h // 4, w // 4
    cropped = img[ch:h-ch, cw:w-cw]
    print(f"Cropped shape: {cropped.shape}")
    
    # Resize (nearest neighbor)
    new_h, new_w = 50, 50
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    resized = img[row_idx][:, col_idx]
    print(f"Resized shape: {resized.shape}")
    
    # Color channel operations
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    print(f"\nRed channel mean: {r.mean():.2f}")
    print(f"Green channel mean: {g.mean():.2f}")
    print(f"Blue channel mean: {b.mean():.2f}")
    
    # Histogram of grayscale
    hist, bins = np.histogram(gray, bins=8, range=(0, 256))
    print(f"\nGrayscale histogram: {hist}")
    
    # Thresholding
    threshold = 128
    binary = (gray > threshold).astype(np.uint8) * 255
    print(f"Binary image (threshold={threshold}): {binary.sum() / 255:.0f} pixels above")
    
    # Invert
    inverted = 255 - img
    print(f"Inverted: mean = {inverted.mean():.2f}")

if __name__ == "__main__":
    img = create_image()
    manipulate(img)