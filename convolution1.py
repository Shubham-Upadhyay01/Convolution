import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Import imageio.v2 to suppress deprecation warning

# Load the image in grayscale mode ('F' for floating point representation)
image = imageio.imread('transparent.png', mode='F')

# Define a 3x3 edge detection kernel
kernel_horizontal = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

kernel_vertical = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])
# Perform the 2D convolution operation
convolved_image1 = convolve2d(image, kernel_horizontal, mode='valid')
convolved_image2 = convolve2d(image, kernel_vertical, mode='valid')

# Display the original and convolved images
plt.figure(figsize=(10, 5))
plt.subplot(311), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(322), plt.imshow(convolved_image1, cmap='gray'), plt.title('Convolved Image 1')
plt.subplot(323), plt.imshow(convolved_image2, cmap='gray'), plt.title('Convolved Image 2')
plt.show()
