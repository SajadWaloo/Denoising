import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.restoration import denoise_wavelet, denoise_nl_means, estimate_sigma

# Function to add Gaussian noise to an image
def add_noise(image, noise_level=0.1):
    return random_noise(image, mode='gaussian', var=noise_level**2)

# Function to display images
def display_images(images, titles, cmap=None):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Load example images
original_image = img_as_float(data.chelsea())
noisy_image = add_noise(original_image)

# Apply denoising techniques
denoised_wavelet = denoise_wavelet(noisy_image, channel_axis=-1)
sigma_est = np.mean(estimate_sigma(noisy_image, channel_axis=-1))
denoised_nl_means = denoise_nl_means(noisy_image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3, channel_axis=-1)

# Display results
images = [original_image, noisy_image, denoised_wavelet, denoised_nl_means]
titles = ['Original Image', 'Noisy Image', 'Denoised with Wavelet', 'Denoised with Non-Local Means']

display_images(images, titles)
