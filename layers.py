import tensorflow as tf

class ConvLayer2D(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super(ConvLayer2D, self).__init__()

        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel,
            strides=stride,
            padding='valid' if padding == 0 else 'same',
            dilation_rate=dilation,
            use_bias=True
        )
        self.drop = tf.keras.layers.Dropout(0.2)

    def call(self, x, training=None, **kwargs):
        x = self.norm(x, training=training)
        x = self.relu(x)
        x = self.conv(x)
        x = self.drop(x, training=training)
        return x
    


class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size):
        super(TemporalBlock, self).__init__()

        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        # Compute padding for each temporal layer to have a fixed-size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size[1] * dilation[1] - 1
            temp_pad = tf.math.floor((filter_size - 1) / 2) - 1 * (dilation[1] // 2 - 1)
            padding.append((0, temp_pad))

        self.layers = [ConvLayer2D(
            in_channels, out_channels, kernel_size, stride, padding[i], dilation_list[i]
        ) for i in range(n_layers)]

    def call(self, x, training=None, **kwargs):
        features = []

        for layer in self.layers:
            out = layer(x, training=training)
            features.append(out)

        out = tf.concat(features, axis=-1)
        return out




class SpatialBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super(SpatialBlock, self).__init__()

        kernel_list = [((input_height // (i + 1)), 1) for i in range(num_spatial_layers)]

        padding = [(
            tf.math.floor((kernel[0] - 1) / 2),
            0
        ) for kernel in kernel_list]

        self.layers = [ConvLayer2D(
            in_channels, out_channels, kernel_list[i], stride, padding[i], 1
        ) for i in range(num_spatial_layers)]

    def call(self, x, training=None, **kwargs):
        features = []

        for layer in self.layers:
            out = layer(x, training=training)
            features.append(out)

        out = tf.concat(features, axis=-1)

        return out


# Function to create a 3x3 convolutional layer
def conv3x3(in_channels, out_channels, stride=1):
    return tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False
    )

# Residual block
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample

    def call(self, x, training=None, **kwargs):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
