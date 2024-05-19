import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Import imageio.v2 to suppress deprecation warning

# Load the image
image = imageio.imread('transparent.png')

# Convert the image to grayscale
image_gray = np.mean(image, axis=2)

# Define a 3x3 edge detection kernel
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Perform the 2D convolution operation
convolved_image = convolve2d(image_gray, kernel, mode='valid')

# Display the original and convolved images
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122), plt.imshow(convolved_image, cmap='gray'), plt.title('Convolved Image')
plt.show()
