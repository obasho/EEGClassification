
import tensorflow as tf
from EEGChannelNet import EegEncoder
from imagenet import ImageEncoder
from utils import triplet_loss



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



