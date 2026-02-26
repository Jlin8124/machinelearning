### Building a simple Vertical Edge Detector

import numpy as np 

# 1. The input "image" (white box on black background)
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=np.float64)

# 2. The Kernel "Weight Matrix" (A 3x3 filter looking for vertical lines)
# This also serves as our weight layer 

kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float64)

# =============================================================================
# 3. Convolution Layer — now with padding, stride, bias, and kernel flipping
# =============================================================================

def convolve2d(img, kernel, padding =0, stride = 1, bias =0.0, flip_kernel=True):
    x = 5


if __name__ == "__main__":
    print("input image (5x5):")
    print (image)
    print()