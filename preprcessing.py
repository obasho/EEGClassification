import tensorflow as tf
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


eeg_signals_path = "eeg_5_95_std.pth"

data=torch.load(eeg_signals_path)

eeg_data_list = []
labels_list = []

for item in data['dataset']:
        # Handle varying types of 'eeg' data (list or ndarray
  eeg_data = np.array(item['eeg'])
  eeg_data_list.append(eeg_data)
  labels_list.append(item['label'])

eeg_data_list=process_arrays(eeg_data_list)

def create_tf_dataset(eeg_data_list, labels_list):

    # Convert labels to one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels_list, num_classes=40)

    # Creating a TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices((eeg_data_list, labels_one_hot))

    return tf_dataset

dataset = create_tf_dataset(np.array(eeg_data_list), labels_list)

def save_tf_dataset(tf_dataset, output_file):
    num_classes = tf_dataset.element_spec[1].shape[-1]  # Get the number of classes from the dataset

    # Define the feature description for serialization
    feature_description = {
        'eeg_data': tf.io.FixedLenFeature([128, 470], tf.float32),
        'label': tf.io.FixedLenFeature([num_classes], tf.float32)
    }

    # Create a TFRecord writer
    with tf.io.TFRecordWriter(output_file) as writer:
        for eeg_data, label in tf_dataset:
            # Serialize the example
            feature = {
                'eeg_data': tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(eeg_data, [-1]))),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())

# Example usage:
output_file = 'eeg_5_95_std.tfrecord'  # Update with your desired output file path
save_tf_dataset(dataset, output_file)