
import tensorflow as tf
from EEGChannelNet import EegEncoder
from imagenet import ImageEncoder
from utils import triplet_loss


in_channels = 128  # Number of input channels (assuming EEG data has 128 channels)
temp_channels = 10  # Number of temporal channels (choose based on your model architecture)
out_channels = 1  # Number of output channels (choose based on your model architecture)
input_width = 470  # Input width (number of time steps in EEG data)
in_height = 128  # Input height (number of EEG channels)
temporal_kernel = 33  # Temporal kernel size (choose based on your model architecture)
temporal_stride = 2  # Temporal stride (choose based on your model architecture)
temporal_dilation_list = [1,2,4,8,16]  # List of temporal dilation rates (choose based on your model architecture)
num_temporal_layers = 5  # Number of temporal layers (choose based on your model architecture)
num_spatial_layers = 4  # Number of spatial layers (choose based on your model architecture)
spatial_stride = 2  # Spatial stride (choose based on your model architecture)
num_residual_blocks = 4  # Number of residual blocks (choose based on your model architecture)
down_kernel = ...  # Downsample kernel size (choose based on your model architecture)
down_stride = ...  # Downsample stride (choose based on your model architecture)
latent_dim = 128  # Dimensionality of the latent space (specified as 128)

# Initialize the EEG encoder

# Define Siamese network
class SiameseNetwork(tf.keras.Model):
    def __init__(self, eeg_encoder, image_encoder):
        super(SiameseNetwork, self).__init__()
        self.eeg_encoder = eeg_encoder
        self.image_encoder = image_encoder

    def call(self, inputs, training=None, **kwargs):
        eeg_input, image_input = inputs
        eeg_embedding = self.eeg_encoder(eeg_input)
        image_embedding = self.image_encoder(image_input)
        return eeg_embedding, image_embedding


# Define the input shapes and other parameters
eeg_input_shape = (128, 470, 1)  # Assuming EEG data shape
image_input_shape = (128, 128, 3)  # Specify image input shape

# Instantiate EEG encoder
eeg_encoder = EegEncoder(in_channels, temp_channels, out_channels, input_width, in_height,
                         temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                         num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride,
                         latent_dim)


# Instantiate image encoder
image_encoder = ImageEncoder(image_input_shape,latent_dim=128)  # Assuming latent dimension is 128

# Instantiate the Siamese model
siamese_model = SiameseNetwork(eeg_encoder, image_encoder)


siamese_model.compile(loss=triplet_loss, optimizer=tf.keras.optimizers.Adam())

# Train the model
siamese_model.fit(triplet_dataset, epochs=num_epochs)
