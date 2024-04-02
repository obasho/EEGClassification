
from EEGChannelNet import EegEncoder
from classifier import EegClassifier

import tensorflow as tf
import torch
import numpy as np
import gc


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


def RetriveData(eeg_signals_path):
    data=torch.load(eeg_signals_path)
    eeg_data_list = []
    Image_list = []

    for item in data['dataset']:
    # Handle varying types of 'eeg' data (list or ndarray
      eeg_data = np.array(item['eeg'])
      eeg_data_list.append(eeg_data)
      Image_list.append(item['image'])
    eeg=process_arrays(eeg_data_list)
    labelone=one_hot_labels = tf.one_hot(Image_list, depth=40)
    input_tensors = [tf.constant(arr, dtype=tf.float32) for arr in eeg]
    label_tensors = [tf.constant(label, dtype=tf.float32) for label in labelone]

    # Step 3: Create a tf.data.Dataset object
    input_dataset = tf.data.Dataset.from_tensor_slices(input_tensors)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_tensors)

    # Step 4: Combine input and label pairs
    dataset = tf.data.Dataset.zip((input_dataset, label_dataset))

    # Optionally, you can prefetch data for better performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


eeg_signals_path = "/content/drive/MyDrive/data/block/eeg_14_70_std.pth"