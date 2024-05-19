import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Import imageio.v2 to suppress deprecation warning

# Load the image
image = imageio.imread('transparent.png')

# Define a 3x3 edge detection kernel
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Initialize an empty array for the convolved image
convolved_image = np.zeros_like(image, dtype=float)

# Perform the 2D convolution operation on each color channel
for c in range(image.shape[2]):
    convolved_image[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same')

# Display the original and convolved images
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122), plt.imshow(convolved_image.astype(np.uint8)), plt.title('Convolved Image')
plt.show()
