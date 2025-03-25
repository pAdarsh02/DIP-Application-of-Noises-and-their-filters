import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import wiener

# Function to load image
def load_image():
    global image
    file_path = filedialog.askopenfilename(title="Select an Image", 
                                           filetypes=[("Image Files", ".jpg;.png;.jpeg;.bmp")])
    if not file_path:
        print("No image selected. Exiting.")
        exit()
    
    image = mpimg.imread(file_path)
    if image.max() <= 1.0:  # Normalize if needed
        image = (image * 255).astype(np.uint8)

# Function to add Gaussian Noise
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Function to add Salt & Pepper Noise
def add_salt_and_pepper_noise(image, prob=0.05):
    noisy = np.copy(image)
    rand_matrix = np.random.rand(*image.shape[:2])
    noisy[rand_matrix < prob] = 0
    noisy[rand_matrix > 1 - prob] = 255
    return noisy

# Function to add Rayleigh Noise
def add_rayleigh_noise(image, scale=30):
    noise = np.random.rayleigh(scale, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Function to add Erlang (Gamma) Noise
def add_erlang_noise(image, shape=2, scale=10):
    noise = np.random.gamma(shape, scale, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Function to add Exponential Noise
def add_exponential_noise(image, scale=30):
    noise = np.random.exponential(scale, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Function to add Uniform Noise
def add_uniform_noise(image, low=-30, high=30):
    noise = np.random.uniform(low, high, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Restoration Filters
def median_filter_restore(image, size=3):
    return median_filter(image, size=size)

def wiener_filter_restore(image, size=3):
    image = image.astype(np.float32) / 255.0  # Normalize

    if len(image.shape) == 3:  # Color image
        restored_image = np.zeros_like(image, dtype=np.float32)
        for i in range(3):  
            restored_channel = wiener(image[:, :, i], (size, size))
            restored_channel = np.nan_to_num(restored_channel)  
            restored_image[:, :, i] = restored_channel
        restored_image = np.clip(restored_image * 255, 0, 255).astype(np.uint8)
    else:  
        restored_image = wiener(image, (size, size))
        restored_image = np.nan_to_num(restored_image)  
        restored_image = np.clip(restored_image * 255, 0, 255).astype(np.uint8)

    return restored_image

def inverse_filter_restore(image, alpha=0.1):
    return np.clip(image / (alpha + 1), 0, 255).astype(np.uint8)

def low_pass_filter_restore(image, size=3):
    return uniform_filter(image, size=size)

# Function to display images
def show_images(noise_type, noisy_image, restored_image, filter_name):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(noisy_image)
    axes[1].set_title(f"{noise_type} Image")
    axes[1].axis("off")

    axes[2].imshow(restored_image)
    axes[2].set_title(f"Restored ({filter_name})")
    axes[2].axis("off")

    plt.show()

# Load Image Before GUI Starts
load_image()

# GUI for the side menu
root = tk.Tk()
root.title("Image Processing - Noise & Restoration")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="W")

ttk.Label(frame, text="Select Noise Type:", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=5)

ttk.Button(frame, text="Gaussian Noise", command=lambda: show_images(
    "Gaussian", add_gaussian_noise(image),
    wiener_filter_restore(add_gaussian_noise(image)), "Wiener Filter")
).grid(row=1, column=0, pady=2)

ttk.Button(frame, text="Salt & Pepper Noise", command=lambda: show_images(
    "Salt & Pepper", add_salt_and_pepper_noise(image),
    median_filter_restore(add_salt_and_pepper_noise(image)), "Median Filter")
).grid(row=2, column=0, pady=2)

ttk.Button(frame, text="Rayleigh Noise", command=lambda: show_images(
    "Rayleigh", add_rayleigh_noise(image),
    median_filter_restore(add_rayleigh_noise(image)), "Adaptive Median Filter")
).grid(row=3, column=0, pady=2)

ttk.Button(frame, text="Erlang (Gamma) Noise", command=lambda: show_images(
    "Erlang (Gamma)", add_erlang_noise(image),
    wiener_filter_restore(add_erlang_noise(image)), "Wiener Filter")
).grid(row=4, column=0, pady=2)

ttk.Button(frame, text="Exponential Noise", command=lambda: show_images(
    "Exponential", add_exponential_noise(image),
    inverse_filter_restore(add_exponential_noise(image)), "Inverse Filtering")
).grid(row=5, column=0, pady=2)

ttk.Button(frame, text="Uniform Noise", command=lambda: show_images(
    "Uniform", add_uniform_noise(image),
    low_pass_filter_restore(add_uniform_noise(image)), "Low-Pass (Mean) Filter")
).grid(row=6, column=0, pady=2)

root.mainloop()