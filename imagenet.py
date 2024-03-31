import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten

class ImageEncoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super(ImageEncoder, self).__init__()

        # Define the layers for image processing
        self.inputs = Input(input_shape)
        self.conv1 = Conv2D(16, 3, activation='relu', padding='same')
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2D(16, 3, activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(32, 3, activation='relu', padding='same')
        self.conv4 = Conv2D(2, 1, activation='relu', padding='same')
        self.batch_norm2 = BatchNormalization()
        self.conv5 = Conv2D(32, 3, activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.conv6 = Conv2D(64, 3, activation='relu', padding='same')
        self.conv7 = Conv2D(2, 1, activation='relu', padding='same')
        self.batch_norm3 = BatchNormalization()
        self.conv8 = Conv2D(64, 3, activation='relu', padding='same')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.conv9 = Conv2D(128, 3, activation='relu', padding='same')
        self.conv10 = Conv2D(4, 1, activation='relu', padding='same')
        self.batch_norm4 = BatchNormalization()
        self.conv11 = Conv2D(1, 3, activation='relu', padding='same')
        self.flatten = Flatten()

        # Define the dense layer for mapping to latent space
        self.dense_layer = tf.keras.layers.Dense(latent_dim)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch_norm2(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.batch_norm3(x)
        x = self.conv8(x)
        x = self.pool3(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.batch_norm4(x)
        x = self.conv11(x)
        x = self.flatten(x)
        latent_space = self.dense_layer(x)

        return latent_space
