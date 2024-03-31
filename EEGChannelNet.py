import tensorflow as tf
from layers import TemporalBlock, SpatialBlock, ResidualBlock, ConvLayer2D

class FeaturesExtractor(tf.keras.layers.Layer):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride):
        super(FeaturesExtractor, self).__init__()

        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temporal_layers, temporal_kernel, temporal_stride, temporal_dilation_list, input_width
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers, out_channels, num_spatial_layers, spatial_stride, in_height
        )

        self.res_blocks = [tf.keras.Sequential([
            ResidualBlock(out_channels * num_spatial_layers, out_channels * num_spatial_layers),
            ConvLayer2D(out_channels * num_spatial_layers, out_channels * num_spatial_layers, down_kernel, down_stride, 0, 1)
        ]) for _ in range(num_residual_blocks)]

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1
        )

    def call(self, x, training=None, **kwargs):
        out = self.temporal_block(x)

        out = self.spatial_block(out)

        if self.res_blocks:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)

        return out

class MyModel(tf.keras.Model):
    def __init__(self, in_channels=1, temp_channels=10, out_channels=50, num_classes=40, embedding_size=1000,
                 input_width=440, input_height=128, temporal_dilation_list=[(1,1),(1,2),(1,4),(1,8),(1,16)],
                 temporal_kernel=(1,33), temporal_stride=(1,2),
                 num_temp_layers=4,
                 num_spatial_layers=4, spatial_stride=(2,1), num_residual_blocks=4, down_kernel=3, down_stride=2):
        super(MyModel, self).__init__()

        self.encoder = FeaturesExtractor(
            in_channels, temp_channels, out_channels, input_width, input_height,
            temporal_kernel, temporal_stride,
            temporal_dilation_list, num_temp_layers,
            num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride
        )

        # Create a dummy input to infer the encoding size
        dummy_input = tf.zeros((1, in_channels, input_height, input_width))
        encoding_size = tf.keras.layers.Flatten()(self.encoder(dummy_input)).shape[-1]

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_size, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, x, training=None, **kwargs):
        x = self.encoder(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.classifier(x)
        return x
    
class EegEncoder(tf.keras.Model):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride,
                 latent_dim):
        super(EegEncoder, self).__init__()

        # FeaturesExtractor as the first layer
        self.features_extractor = FeaturesExtractor(
            in_channels, temp_channels, out_channels, input_width, in_height,
            temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
            num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride
        )

        # Dense layer to map to latent space
        self.dense_layer = tf.keras.layers.Dense(latent_dim)

    def call(self, x, training=None, **kwargs):
        # Pass input through FeaturesExtractor
        features = self.features_extractor(x)

        # Flatten the features
        features_flattened = tf.keras.layers.Flatten()(features)

        # Map to latent space
        latent_space = self.dense_layer(features_flattened)

        return latent_space
