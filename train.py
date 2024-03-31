import tensorflow as tf
import torch
import numpy as np
from datapipeline import map_label_to_image_path
from datapipeline import load_and_preprocess_image
from datapipeline import generate_triplets

import tensorflow as tf
from EEGChannelNet import EegEncoder
from imagenet import ImageEncoder
from utils import triplet_loss
from siamesenetwork import SiameseNetwork


in_channels = 128  # Number of input channels (assuming EEG data has 128 channels)
temp_channels = 10  # Number of temporal channels 
out_channels = 1  # Number of output channels 
input_width = 470  # Input width (number of time steps in EEG data)
in_height = 128  # Input height (number of EEG channels)
temporal_kernel = 33  # Temporal kernel size 
temporal_stride = 2  # Temporal stride 
temporal_dilation_list = [1,2,4,8,16]  # List of temporal dilation rates 
num_temporal_layers = 5  # Number of temporal layers 
num_spatial_layers = 4  # Number of spatial layers 
spatial_stride = 2  # Spatial stride 
num_residual_blocks = 4  # Number of residual blocks 
down_kernel = (3,3) # Downsample kernel size 
down_stride = 1  # Downsample stride 
latent_dim = 128  # Dimensionality of the latent space (specified as 128)

num_epochs=100



eeg_signals_path = "/home/obasho/Documents/cs671project/data/block/eeg_5_95_std.pth"
data=torch.load(eeg_signals_path)

image_names=data['images']
dataset=data['dataset']
class_names=data['labels']
eeg_data = []
image_labels = []

for item in data['dataset']:
    eegd=np.array(item['eeg'])
    eeg_data.append(eegd)
    image_labels.append(item['label'])

# Step 1: Create dataset for EEG data and image labels
eeg_dataset = tf.data.Dataset.from_tensor_slices(eeg_data)
label_dataset = tf.data.Dataset.from_tensor_slices(image_labels)

image_paths_dataset = label_dataset.map(lambda label: tf.py_function(func=map_label_to_image_path, inp=[label], Tout=tf.string))
image_dataset = image_paths_dataset.map(load_and_preprocess_image)


triplet_dataset = tf.data.Dataset.zip((eeg_dataset, image_dataset, label_dataset))
triplet_dataset = triplet_dataset.map(generate_triplets)


# Step 5: Preprocess and shuffle the dataset
triplet_dataset = triplet_dataset.shuffle(buffer_size=len(eeg_data))

# Step 6: Batch and prefetch the dataset
batch_size = 32
triplet_dataset = triplet_dataset.batch(batch_size)
triplet_dataset = triplet_dataset.prefetch(tf.data.experimental.AUTOTUNE)

eeg_input_shape = (128, 470, 1)  # Assuming EEG data shape
image_input_shape = (128, 128, 3)  # Specify image input shape
# Instantiate EEG encoder
eeg_encoder = EegEncoder(in_channels, temp_channels, out_channels, input_width, in_height,
                         temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                         num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride,
                         latent_dim)


image_encoder = ImageEncoder(image_input_shape,latent_dim=128)  # Assuming latent dimension is 128

# Instantiate the Siamese model
siamese_model = SiameseNetwork(eeg_encoder, image_encoder)


siamese_model.compile(loss=triplet_loss, optimizer=tf.keras.optimizers.Adam())

# Train the model
siamese_model.fit(triplet_dataset, epochs=num_epochs)
