import os.path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch

from constants import *

def get_image_path(file_name):
    return f'{IMAGES_FOLDER}/{file_name}.jpg'

def does_image_exists(file_name):
    return os.path.isfile(get_image_path(file_name))

def plot_images(df, r, c):
    first_files = df['file_name'].head(r*c)
    body_parts = df['body_parts'].head(r*c)
    fig, axes = plt.subplots(r, c, figsize=(int(15/4*c), int(10/3*r)))
    for ax, file, body_part in zip(axes.flatten(), first_files, body_parts):
        try:
            # Read and show the image from the file path
            img = mpimg.imread(get_image_path(file))
            ax.imshow(img)
        except Exception as e:
            # In case the image fails to load, display an error message on the subplot
            ax.text(0.5, 0.5, 'Image not found', horizontalalignment='center', verticalalignment='center')
        ax.set_title(file+'-'+body_part)
        ax.axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.show()

def get_torch_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device