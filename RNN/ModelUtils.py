import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from DataHandler import DataHandler
import os
class ModelUtils:
    def create_checkpoint_callback(self, checkpoint_dir='checkpoints'):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{checkpoint_dir}/ckpt_{{epoch}}.weights.h5',
            save_weights_only=True
        )
    
    def train_model(self, model, dataset, epochs, checkpoint_subdir):
        full_path = os.path.join('checkpoints', checkpoint_subdir)
        os.makedirs(full_path, exist_ok=True)
        checkpoint_callback = self.create_checkpoint_callback(full_path)
        training_history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
        return training_history
    
    def generate_text(self,
        model,
        checkpoint_dir,
        char2idx,
        idx2char,
        seed="ROMEO: ",
        gen_length=500,
        temperature=1.0
    ):
        import glob
        import os
        # Load weights from the latest checkpoint
        weight_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.weights.h5")))
        latest_ckpt = weight_files[-1]
        model.load_weights(latest_ckpt)
        model.build(tf.TensorShape([1, None]))  # Dynamically shaped input

        # Convert seed string to indices
        input_indices = [char2idx[c] for c in seed]
        input_tensor = tf.expand_dims(input_indices, axis=0)  # Shape (1, len)

        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

        generated = list(seed)

        for _ in range(gen_length):
            # Predict logits for next character
            predictions = model(input_tensor)
            logits = predictions[:, -1, :] / temperature  # Last timestep

            # Sample from logits
            next_id = tf.random.categorical(logits, num_samples=1)[0, 0].numpy()

            # Append predicted char and update input
            generated.append(idx2char[next_id])
            input_tensor = tf.expand_dims([next_id], axis=0)

        return ''.join(generated)