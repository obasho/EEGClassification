import tensorflow as tf
import numpy as np
import os
import torch
import numpy as np

def process_arrays(list_of_arrays):
    processed_arrays = []

    for arr in list_of_arrays:
        # Remove the first 20 data points from each channel
        processed_arr = arr[:, 20:]

        # If the array has more than 470 columns, truncate to 470
        if processed_arr.shape[1] > 470:
            processed_arr = processed_arr[:, :470]

        # If the array has fewer than 470 columns, pad with zeros
        elif processed_arr.shape[1] < 470:
            padding = np.zeros((128, 470 - processed_arr.shape[1]))
            processed_arr = np.concatenate([processed_arr, padding], axis=1)

        processed_arrays.append(processed_arr)

    return processed_arrays


# Parameters
target_height = 128  # Specify target height for image resizing
target_width = 128  # Specify target width for image resizing





# Step 3: Load and preprocess images
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Adjust channels if necessary
    image = tf.image.resize(image, (target_height, target_width))  # Adjust target height and width
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image


# Step 4: Combine EEG data and images into triplets
def generate_triplets(eeg_data, images, labels):
    anchor_eeg = eeg_data
    anchor_label = labels
    anchor_image = images

    # Find positive samples (samples with the same label as the anchor)
    positive_mask = tf.equal(labels, anchor_label)
    positive_indices = tf.where(positive_mask)[:, 0]
    positive_index = tf.random.shuffle(positive_indices)[0]
    positive_image = tf.gather(images, positive_index)

    # Find negative samples (samples with a different label from the anchor)
    negative_mask = tf.logical_not(positive_mask)
    negative_indices = tf.where(negative_mask)[:, 0]
    negative_index = tf.random.shuffle(negative_indices)[0]
    negative_image = tf.gather(images, negative_index)

    return anchor_eeg, anchor_image, positive_image, negative_image

