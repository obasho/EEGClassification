import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128, num_classes=40):
        super(Model, self).__init__()

        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.num_classes = num_classes

        # Define internal modules
        self.lstm = tf.keras.layers.LSTM(
            units=lstm_size,
            return_sequences=False,
            return_state=False,
            stateful=False,
            batch_input_shape=(None, None, input_size),
            dropout=0.0,
            recurrent_dropout=0.0,
            recurrent_initializer='glorot_uniform'
        )

        self.output_layer = tf.keras.layers.Dense(output_size, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=None, **kwargs):
        # Prepare LSTM initial state
        batch_size = tf.shape(x)[0]
        lstm_init = [tf.zeros((batch_size, self.lstm_size)), tf.zeros((batch_size, self.lstm_size))]

        # Forward LSTM and get final state
        x = self.lstm(x, initial_state=lstm_init)

        # Forward output
        x = tf.nn.relu(self.output_layer(x))
        x = self.classifier(x)

        return x
