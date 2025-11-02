  -----------------  Q1  -------------------

# import cv2
# import numpy as np

# # Load the image
# horse = cv2.imread("/content/horse.png", cv2.IMREAD_GRAYSCALE)

# # Check if all pixel values are 0
# if np.all(horse == 0):
#     print("All pixel values are 0.")
# else:
#     print("Not all pixel values are 0.")

# print("Pixel at (150, 100):", horse[200, 90])
# print("Pixel at (180, 200):", horse[10, 10])



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Manual queue implementation to replace collections.deque
class ManualQueue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def is_empty(self):
        return len(self.items) == 0

# Manual thresholding implementation
def manual_threshold(image, threshold=127):
    """Implement thresholding manually without using cv2.threshold()"""
    result = np.zeros_like(image)
    h, w = image.shape
    
    for i in range(h):
        for j in range(w):
            if image[i, j] > threshold:
                result[i, j] = 1
            else:
                result[i, j] = 0
    
    return result

# Adjacency functions
def neighbors_4(x, y, shape):
    """Return 4-adjacency neighbors"""
    h, w = shape
    neighbors = []
    if x > 0: neighbors.append((x-1, y))     # up
    if x < h-1: neighbors.append((x+1, y))   # down
    if y > 0: neighbors.append((x, y-1))     # left
    if y < w-1: neighbors.append((x, y+1))   # right
    return neighbors

def neighbors_8(x, y, shape):
    """Return 8-adjacency neighbors"""
    h, w = shape
    nbs = neighbors_4(x, y, (h, w))
    if x > 0 and y > 0: nbs.append((x-1, y-1))   # top-left
    if x > 0 and y < w-1: nbs.append((x-1, y+1)) # top-right
    if x < h-1 and y > 0: nbs.append((x+1, y-1)) # bottom-left
    if x < h-1 and y < w-1: nbs.append((x+1, y+1)) # bottom-right
    return nbs

def neighbors_m(x, y, img):
    """Return m-adjacency neighbors with proper condition checking"""
    h, w = img.shape
    nbs = neighbors_4(x, y, (h, w))
    
    # Check diagonal neighbors with m-adjacency condition
    diagonals = []
    if x > 0 and y > 0: diagonals.append((x-1, y-1))   # top-left
    if x > 0 and y < w-1: diagonals.append((x-1, y+1)) # top-right
    if x < h-1 and y > 0: diagonals.append((x+1, y-1)) # bottom-left
    if x < h-1 and y < w-1: diagonals.append((x+1, y+1)) # bottom-right
    
    for nx, ny in diagonals:
        # For m-adjacency, diagonal neighbors are included only if 
        # the corresponding horizontal and vertical neighbors are both 0
        if (x == nx):  # Same row
            if img[x, min(y, ny)+1] == 0:  # Check the pixel between them
                nbs.append((nx, ny))
        elif (y == ny):  # Same column
            if img[min(x, nx)+1, y] == 0:  # Check the pixel between them
                nbs.append((nx, ny))
        else:
            # For diagonal, check if both adjacent 4-neighbors are 0
            if img[x, ny] == 0 and img[nx, y] == 0:
                nbs.append((nx, ny))
    
    return nbs

def is_connected(img, start, end, mode="4"):
    """
    BFS to check if 'start' and 'end' pixels are connected 
    under given adjacency mode.
    """
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)
    q = ManualQueue()
    q.enqueue(start)
    visited[start] = True

    while not q.is_empty():
        cx, cy = q.dequeue()
        if (cx, cy) == end:
            return True
            
        if mode == "4":
            neighbors = neighbors_4(cx, cy, (h, w))
        elif mode == "8":
            neighbors = neighbors_8(cx, cy, (h, w))
        elif mode == "m":
            neighbors = neighbors_m(cx, cy, img)
        else:
            raise ValueError("Mode must be '4', '8', or 'm'")
            
        for nx, ny in neighbors:
            if 0 <= nx < h and 0 <= ny < w and img[nx, ny] == 1 and not visited[nx, ny]:
                visited[nx, ny] = True
                q.enqueue((nx, ny))
                
    return False

# -----------------------------
# Load and Preprocess Images
# -----------------------------

# Load Cameraman
cameraman = cv2.imread("/content/cameraman.png", cv2.IMREAD_GRAYSCALE)
# Manual thresholding
cam_bin = manual_threshold(cameraman, 127)

# Load Horse
horse = cv2.imread("/content/horse.png", cv2.IMREAD_GRAYSCALE)
# Manual thresholding with inversion (horse is black on white background)
horse_bin = manual_threshold(horse, 127)
horse_bin = 1 - horse_bin  # Invert manually

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(cam_bin, cmap='gray')
plt.title("Cameraman Binary")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(horse_bin, cmap='gray')
plt.title("Horse Binary (processed)")
plt.axis("off")

plt.show()

# -----------------------------
# Connectivity Tests
# -----------------------------
# Pick some points inside images (adjust as needed)
cam_start, cam_end = (100,100), (120,120)
horse_start, horse_end = (150,100), (180,200)

print("\nCameraman Connectivity:")
print("4-adj:", is_connected(cam_bin, cam_start, cam_end, "4"))
print("8-adj:", is_connected(cam_bin, cam_start, cam_end, "8"))
print("m-adj:", is_connected(cam_bin, cam_start, cam_end, "m"))

print("\nHorse Connectivity:")
print("4-adj:", is_connected(horse_bin, horse_start, horse_end, "4"))
print("8-adj:", is_connected(horse_bin, horse_start, horse_end, "8"))
print("m-adj:", is_connected(horse_bin, horse_start, horse_end, "m"))




-------------------------------------- Q2 -------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def log_transform(I, c=1.0):
    """
    Apply logarithmic transform: J = c * log(1 + I)
    Manual implementation without high-level functions
    """
    rows, cols = I.shape
    result = np.zeros_like(I, dtype=np.float64)
    
    # Manual logarithmic transformation
    for i in range(rows):
        for j in range(cols):
            # Normalize to [0,1] and apply log transform
            normalized_pixel = I[i, j] / 255.0
            result[i, j] = c * math.log(1 + normalized_pixel)
    
    # Scale back to 0-255 range and convert to uint8
    output = np.zeros_like(I)
    for i in range(rows):
        for j in range(cols):
            output[i, j] = np.clip(result[i, j] * 255, 0, 255).astype(np.uint8)
    
    return output

def gamma_transform(I, gamma=1.0, c=1.0):
    """
    Apply power-law (gamma) transform: J = c * I^gamma
    Manual implementation without high-level functions
    """
    rows, cols = I.shape
    result = np.zeros_like(I, dtype=np.float64)
    
    # Manual gamma transformation
    for i in range(rows):
        for j in range(cols):
            # Normalize to [0,1] and apply gamma transform
            normalized_pixel = I[i, j] / 255.0
            result[i, j] = c * (normalized_pixel ** gamma)
    
    # Scale back to 0-255 range and convert to uint8
    output = np.zeros_like(I)
    for i in range(rows):
        for j in range(cols):
            output[i, j] = np.clip(result[i, j] * 255, 0, 255).astype(np.uint8)
    
    return output

def contrast_stretch(I, r1, s1, r2, s2):
    """
    Apply piecewise linear contrast stretching
    Manual implementation without high-level functions
    """
    rows, cols = I.shape
    result = np.zeros_like(I)
    
    # Calculate slopes for piecewise linear function
    slope1 = s1 / r1 if r1 > 0 else 0
    slope2 = (s2 - s1) / (r2 - r1) if r2 != r1 else 0
    slope3 = (255 - s2) / (255 - r2) if r2 < 255 else 0
    
    # Manual contrast stretching
    for i in range(rows):
        for j in range(cols):
            pixel = I[i, j]
            
            if pixel <= r1:
                result[i, j] = slope1 * pixel
            elif pixel <= r2:
                result[i, j] = s1 + slope2 * (pixel - r1)
            else:
                result[i, j] = s2 + slope3 * (pixel - r2)
    
    return result.astype(np.uint8)

def main():
    # Load specific images as required - remove the /content/ prefix if files are in current directory
    coins = cv2.imread("coins.png", cv2.IMREAD_GRAYSCALE)
    astronaut = cv2.imread("astronaut.png", cv2.IMREAD_GRAYSCALE)
    
    if coins is None:
        print("Error: Could not load coins.png")
        print("Please ensure coins.png is in the current directory")
        return
    if astronaut is None:
        print("Error: Could not load astronaut.png")
        print("Please ensure astronaut.png is in the current directory")
        return
    
    print(f"Coins image size: {coins.shape}")
    print(f"Astronaut image size: {astronaut.shape}")
    
    # Apply transformations as specified:
    # coins.png for log & contrast stretching
    # astronaut.png for gamma/power-law
    
    # 1. Logarithmic Transformation on coins.png
    print("Applying logarithmic transformation on coins.png...")
    coins_log = log_transform(coins, c=1.5)
    
    # 2. Contrast Stretching on coins.png
    print("Applying contrast stretching on coins.png...")
    coins_stretched = contrast_stretch(coins, r1=50, s1=10, r2=200, s2=245)
    
    # 3. Gamma Transformation on astronaut.png
    print("Applying gamma transformation on astronaut.png...")
    astronaut_gamma_04 = gamma_transform(astronaut, gamma=0.4, c=1.0)  # Enhance dark areas
    astronaut_gamma_15 = gamma_transform(astronaut, gamma=1.5, c=1.0)  # Enhance bright areas
    astronaut_gamma_25 = gamma_transform(astronaut, gamma=2.5, c=1.0)  # Strong contrast
    
    # Create visualization
    plt.figure(figsize=(16, 10))
    
    # Row 1: coins.png transformations
    plt.subplot(2, 4, 1)
    plt.imshow(coins, cmap='gray')
    plt.title('Coins Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(coins_log, cmap='gray')
    plt.title('Coins: Log Transform\n(c=1.5)')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(coins_stretched, cmap='gray')
    plt.title('Coins: Contrast Stretch\n[50,200]→[10,245]')
    plt.axis('off')
    
    # Empty space for alignment
    plt.subplot(2, 4, 4)
    plt.axis('off')
    
    # Row 2: astronaut.png transformations
    plt.subplot(2, 4, 5)
    plt.imshow(astronaut, cmap='gray')
    plt.title('Astronaut Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(astronaut_gamma_04, cmap='gray')
    plt.title('Astronaut: Gamma=0.4\n(Enhance dark)')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(astronaut_gamma_15, cmap='gray')
    plt.title('Astronaut: Gamma=1.5\n(Enhance bright)')
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(astronaut_gamma_25, cmap='gray')
    plt.title('Astronaut: Gamma=2.5\n(Strong contrast)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("All point-based transformations completed successfully!")
    print("✓ Logarithmic transform applied to coins.png")
    print("✓ Contrast stretching applied to coins.png") 
    print("✓ Gamma transform applied to astronaut.png")

if __name__ == "__main__":
    main()



---------------------------------------------- q3 -------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Manual Histogram, PDF, and CDF Computation
# -----------------------------
def compute_histogram(img):
    """Compute histogram manually (0-255)."""
    h = np.zeros(256, dtype=int)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            h[img[i, j]] += 1
    return h

def compute_pdf(hist, total_pixels):
    """Compute Probability Density Function manually."""
    pdf = np.zeros(256, dtype=float)
    for i in range(256):
        pdf[i] = hist[i] / total_pixels
    return pdf

def compute_cdf(pdf):
    """Compute Cumulative Distribution Function manually."""
    cdf = np.zeros(256, dtype=float)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    return cdf

# -----------------------------
# Histogram Equalization
# -----------------------------
def global_hist_equalization(img):
    """Global histogram equalization implemented manually."""
    rows, cols = img.shape
    total_pixels = rows * cols
    
    # Compute histogram, PDF, and CDF manually
    hist = compute_histogram(img)
    pdf = compute_pdf(hist, total_pixels)
    cdf = compute_cdf(pdf)
    
    # Create mapping from CDF
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = int(255 * cdf[i] + 0.5)
    
    # Apply mapping
    out = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = mapping[img[i, j]]
    
    return out

# -----------------------------
# Local Histogram Equalization (Simplified)
# -----------------------------
def local_hist_equalization(img, window_size=15):
    """Local histogram equalization with manual implementation."""
    rows, cols = img.shape
    out = np.zeros_like(img)
    half_window = window_size // 2
    
    # Create padded image for border handling
    padded_img = np.pad(img, half_window, mode='reflect')
    
    for i in range(rows):
        for j in range(cols):
            # Extract window from padded image
            window = padded_img[i:i+window_size, j:j+window_size]
            
            # Compute histogram manually
            hist = np.zeros(256, dtype=int)
            window_flat = window.flatten()
            for pixel_val in window_flat:
                hist[pixel_val] += 1
            
            # Compute CDF directly from histogram
            cdf = np.zeros(256, dtype=float)
            cdf[0] = hist[0]
            for k in range(1, 256):
                cdf[k] = cdf[k-1] + hist[k]
            
            # Normalize CDF
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]
            
            # Map the center pixel
            center_pixel = img[i, j]
            out[i, j] = int(255 * cdf[center_pixel] + 0.5)
    
    return out

# -----------------------------
# Histogram Specification (Matching)
# -----------------------------
def match_histogram(src, ref):
    """Histogram matching implemented manually."""
    # Compute dimensions
    src_total_pixels = src.size
    ref_total_pixels = ref.size
    
    # Compute histograms, PDFs, and CDFs manually
    src_hist = compute_histogram(src)
    src_pdf = compute_pdf(src_hist, src_total_pixels)
    src_cdf = compute_cdf(src_pdf)
    
    ref_hist = compute_histogram(ref)
    ref_pdf = compute_pdf(ref_hist, ref_total_pixels)
    ref_cdf = compute_cdf(ref_pdf)
    
    # Build mapping manually
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        min_diff = float('inf')
        best_match = 0
        for j in range(256):
            diff = abs(src_cdf[i] - ref_cdf[j])
            if diff < min_diff:
                min_diff = diff
                best_match = j
        mapping[i] = best_match
    
    # Apply mapping
    matched = np.zeros_like(src)
    rows, cols = src.shape
    for i in range(rows):
        for j in range(cols):
            matched[i, j] = mapping[src[i, j]]
    
    return matched


# -----------------------------
# Main Execution
# -----------------------------
def main():
    # Load images
    try:
        xray = cv2.imread("chest_xray.png", cv2.IMREAD_GRAYSCALE)
        coffee = cv2.imread("coffee.png", cv2.IMREAD_GRAYSCALE)
        
        if xray is None:
            print("Error: Could not load chest_xray.png")
            return
        if coffee is None:
            print("Error: Could not load coffee.png")
            return
            
        print("Images loaded successfully")
        
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Apply processing
    print("Applying global histogram equalization...")
    xray_global = global_hist_equalization(xray)
    coffee_global = global_hist_equalization(coffee)
    
    print("Applying local histogram equalization...")
    xray_local = local_hist_equalization(xray)
    coffee_local = local_hist_equalization(coffee)
    
    print("Applying histogram matching...")
    xray_matched = match_histogram(xray, coffee)
    
    print("Processing complete. Displaying results...")
    
    # Simple visualization without histograms first
    plt.figure(figsize=(15, 10))
    
    # Row 1: Original images
    plt.subplot(3, 3, 1)
    plt.imshow(xray, cmap='gray')
    plt.title("X-ray Original")
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(xray_global, cmap='gray')
    plt.title("X-ray Global Equalized")
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(xray_local, cmap='gray')
    plt.title("X-ray Local Equalized")
    plt.axis('off')
    
    # Row 2: Coffee images
    plt.subplot(3, 3, 4)
    plt.imshow(coffee, cmap='gray')
    plt.title("Coffee Original")
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(coffee_global, cmap='gray')
    plt.title("Coffee Global Equalized")
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(coffee_local, cmap='gray')
    plt.title("Coffee Local Equalized")
    plt.axis('off')
    
    # Row 3: Histogram matching results
    plt.subplot(3, 3, 7)
    plt.imshow(xray, cmap='gray')
    plt.title("X-ray Source")
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(coffee, cmap='gray')
    plt.title("Coffee Reference")
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(xray_matched, cmap='gray')
    plt.title("X-ray Matched to Coffee")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

-----------------------------------------------------  q4 ------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_level_slicing_preserve(img, lower_bound, upper_bound, highlight_intensity=255):
    """
    Gray-level slicing with background preserved.
    Highlights specified intensity range while preserving other intensities.
    
    Parameters:
    img: input grayscale image (numpy array)
    lower_bound: lower intensity bound to highlight (100)
    upper_bound: upper intensity bound to highlight (180)
    highlight_intensity: intensity value for highlighting (255)
    
    Returns:
    processed image with background preserved
    """
    rows, cols = img.shape
    result = np.zeros_like(img)
    
    for i in range(rows):
        for j in range(cols):
            pixel_value = img[i, j]
            if lower_bound <= pixel_value <= upper_bound:
                result[i, j] = highlight_intensity
            else:
                result[i, j] = pixel_value  # Preserve original intensity
    
    return result

def gray_level_slicing_suppress(img, lower_bound, upper_bound, highlight_intensity=255):
    """
    Gray-level slicing with background suppressed.
    Highlights specified intensity range and suppresses others to 0.
    
    Parameters:
    img: input grayscale image (numpy array)
    lower_bound: lower intensity bound to highlight (100)
    upper_bound: upper intensity bound to highlight (180)
    highlight_intensity: intensity value for highlighting (255)
    
    Returns:
    processed image with background suppressed
    """
    rows, cols = img.shape
    result = np.zeros_like(img)
    
    for i in range(rows):
        for j in range(cols):
            pixel_value = img[i, j]
            if lower_bound <= pixel_value <= upper_bound:
                result[i, j] = highlight_intensity
            else:
                result[i, j] = 0  # Suppress background to 0
    
    return result

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_level_slicing_preserve(img, lower_bound, upper_bound, highlight_intensity=255):
    """
    Gray-level slicing with background preserved.
    Highlights specified intensity range while preserving other intensities.
    
    Parameters:
    img: input grayscale image (numpy array)
    lower_bound: lower intensity bound to highlight (100)
    upper_bound: upper intensity bound to highlight (180)
    highlight_intensity: intensity value for highlighting (255)
    
    Returns:
    processed image with background preserved
    """
    rows, cols = img.shape
    result = np.zeros_like(img)
    
    for i in range(rows):
        for j in range(cols):
            pixel_value = img[i, j]
            if lower_bound <= pixel_value <= upper_bound:
                result[i, j] = highlight_intensity
            else:
                result[i, j] = pixel_value  # Preserve original intensity
    
    return result

def gray_level_slicing_suppress(img, lower_bound, upper_bound, highlight_intensity=255):
    """
    Gray-level slicing with background suppressed.
    Highlights specified intensity range and suppresses others to 0.
    
    Parameters:
    img: input grayscale image (numpy array)
    lower_bound: lower intensity bound to highlight (100)
    upper_bound: upper intensity bound to highlight (180)
    highlight_intensity: intensity value for highlighting (255)
    
    Returns:
    processed image with background suppressed
    """
    rows, cols = img.shape
    result = np.zeros_like(img)
    
    for i in range(rows):
        for j in range(cols):
            pixel_value = img[i, j]
            if lower_bound <= pixel_value <= upper_bound:
                result[i, j] = highlight_intensity
            else:
                result[i, j] = 0  # Suppress background to 0
    
    return result

def main():
    # Load the satellite image
    satellite = cv2.imread("satellite.png", cv2.IMREAD_GRAYSCALE)
    
    if satellite is None:
        print("Error: Could not load satellite.png")
        return
    
    # Define intensity range to highlight (100-180 as specified)
    LOWER_BOUND = 100
    UPPER_BOUND = 180
    HIGHLIGHT_INTENSITY = 255
    
    # Apply gray-level slicing
    preserved_img = gray_level_slicing_preserve(satellite, LOWER_BOUND, UPPER_BOUND, HIGHLIGHT_INTENSITY)
    suppressed_img = gray_level_slicing_suppress(satellite, LOWER_BOUND, UPPER_BOUND, HIGHLIGHT_INTENSITY)
    
    # Create simple visualization
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(satellite, cmap='gray')
    plt.title('Original Satellite Image')
    plt.axis('off')
    
    # Background preserved
    plt.subplot(1, 3, 2)
    plt.imshow(preserved_img, cmap='gray')
    plt.title(f'Background Preserved\nRange: {LOWER_BOUND}-{UPPER_BOUND}')
    plt.axis('off')
    
    # Background suppressed
    plt.subplot(1, 3, 3)
    plt.imshow(suppressed_img, cmap='gray')
    plt.title(f'Background Suppressed\nRange: {LOWER_BOUND}-{UPPER_BOUND}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print basic information
    print("Gray-Level Slicing Complete")
    print(f"Highlighted intensity range: {LOWER_BOUND}-{UPPER_BOUND}")
    print(f"Original image size: {satellite.shape}")

if __name__ == "__main__":
    main()

-------------------------------------------- q5 --------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

def global_thresholding(img, threshold_value=127):
    """
    Global thresholding implementation.
    
    Parameters:
    img: input grayscale image
    threshold_value: threshold value (default: 127)
    
    Returns:
    binary image
    """
    rows, cols = img.shape
    binary_img = np.zeros_like(img)
    
    for i in range(rows):
        for j in range(cols):
            if img[i, j] >= threshold_value:
                binary_img[i, j] = 255  # Foreground
            else:
                binary_img[i, j] = 0    # Background
    
    return binary_img

def adaptive_thresholding_mean(img, block_size=11, C=5):
    """
    Adaptive thresholding using local mean.
    
    Parameters:
    img: input grayscale image
    block_size: size of local neighborhood (must be odd)
    C: constant subtracted from mean
    
    Returns:
    binary image
    """
    rows, cols = img.shape
    binary_img = np.zeros_like(img)
    half_size = block_size // 2
    
    # Add padding to handle borders
    padded_img = np.pad(img, half_size, mode='reflect')
    
    for i in range(rows):
        for j in range(cols):
            # Extract local neighborhood from padded image
            window = padded_img[i:i+block_size, j:j+block_size]
            
            # Calculate local mean manually with float to prevent overflow
            local_sum = 0.0
            for x in range(block_size):
                for y in range(block_size):
                    local_sum += float(window[x, y])
            local_mean = local_sum / (block_size * block_size)
            
            # Apply threshold
            threshold = local_mean - C
            if img[i, j] >= threshold:
                binary_img[i, j] = 255
            else:
                binary_img[i, j] = 0
    
    return binary_img

def adaptive_thresholding_median(img, block_size=11, C=5):
    """
    Adaptive thresholding using local median.
    
    Parameters:
    img: input grayscale image
    block_size: size of local neighborhood (must be odd)
    C: constant subtracted from median
    
    Returns:
    binary image
    """
    rows, cols = img.shape
    binary_img = np.zeros_like(img)
    half_size = block_size // 2
    
    # Add padding to handle borders
    padded_img = np.pad(img, half_size, mode='reflect')
    
    for i in range(rows):
        for j in range(cols):
            # Extract local neighborhood from padded image
            window = padded_img[i:i+block_size, j:j+block_size]
            
            # Convert to list for sorting
            pixels = []
            for x in range(block_size):
                for y in range(block_size):
                    pixels.append(window[x, y])
            
            # Sort using built-in sort (manual sort is too slow for this)
            pixels.sort()
            local_median = pixels[len(pixels) // 2]
            
            # Apply threshold
            threshold = local_median - C
            if img[i, j] >= threshold:
                binary_img[i, j] = 255
            else:
                binary_img[i, j] = 0
    
    return binary_img



def main():
    # Load images
    coins = cv2.imread("coins.png", cv2.IMREAD_GRAYSCALE)
    text = cv2.imread("text.png", cv2.IMREAD_GRAYSCALE)
    
    if coins is None or text is None:
        print("Error: Could not load images")
        return
    
    print(f"Coins image size: {coins.shape}")
    print(f"Text image size: {text.shape}")
    
    # Apply global thresholding
    print("Applying global thresholding...")
    coins_global = global_thresholding(coins, 127)
    text_global = global_thresholding(text, 127)
    
    # Apply adaptive thresholding (mean)
    print("Applying adaptive thresholding (mean)...")
    coins_adaptive_mean = adaptive_thresholding_mean(coins, block_size=11, C=5)
    text_adaptive_mean = adaptive_thresholding_mean(text, block_size=11, C=5)
    
    # Apply adaptive thresholding (median)
    print("Applying adaptive thresholding (median)...")
    coins_adaptive_median = adaptive_thresholding_median(coins, block_size=11, C=5)
    text_adaptive_median = adaptive_thresholding_median(text, block_size=11, C=5)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Coins images
    plt.subplot(2, 4, 1)
    plt.imshow(coins, cmap='gray')
    plt.title('Coins Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(coins_global, cmap='gray')
    plt.title('Coins Global (T=127)')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(coins_adaptive_mean, cmap='gray')
    plt.title('Coins Adaptive Mean')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(coins_adaptive_median, cmap='gray')
    plt.title('Coins Adaptive Median')
    plt.axis('off')
    
    # Text images
    plt.subplot(2, 4, 5)
    plt.imshow(text, cmap='gray')
    plt.title('Text Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(text_global, cmap='gray')
    plt.title('Text Global (T=127)')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(text_adaptive_mean, cmap='gray')
    plt.title('Text Adaptive Mean')
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(text_adaptive_median, cmap='gray')
    plt.title('Text Adaptive Median')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Thresholding complete!")

if __name__ == "__main__":
    main()

----------------------------------------------------------------------------------------

