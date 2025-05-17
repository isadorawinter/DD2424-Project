import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from DataHandler import DataHandler
class LSTM:
    def build_model(self, batch_size, sequence_length, vocab_size, embedding_dim, n_rnn_units, learning_rate, num_layers):
        # Creates a Keras Sequential model,meaning the layers are stacked
        # from input to output.
        inputs = tf.keras.Input(
            batch_shape=(batch_size, sequence_length), dtype=tf.int32
        )

        x = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )(inputs)
        
        for _ in range(num_layers):
            x = tf.keras.layers.LSTM(
                units=n_rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'
            )(x)

        outputs = tf.keras.layers.Dense(vocab_size)(x)

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

        model.summary()
        return model