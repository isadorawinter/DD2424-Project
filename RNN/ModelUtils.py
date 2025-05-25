import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
import matplotlib as plt
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
        temperature=1.0,
        top_n=5
    ):
        import glob
        import os

        weight_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.weights.h5")))
        latest_ckpt = weight_files[-1]
        model.load_weights(latest_ckpt)
        model.build(tf.TensorShape([1, None]))  

        
        input_indices = [char2idx[c] for c in seed]
        input_tensor = tf.expand_dims(input_indices, axis=0)  

        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

        generated = list(seed)

        for _ in range(gen_length):
           
            predictions = model(input_tensor)
            logits = predictions[:, -1, :] / temperature 

            next_id = tf.random.categorical(logits, num_samples=1)[0, 0].numpy()

            generated.append(idx2char[next_id])
            input_tensor = tf.expand_dims([next_id], axis=0)

        return ''.join(generated)
    

    def visualizing_predictions(self,
        model,
        checkpoint_dir,
        char2idx,
        idx2char,
        seed="ROMEO: ",
        gen_length=500,
        temperature=1.0,
        top_n=5
    ):
        import glob
        import os
      
        weight_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.weights.h5")))
        latest_ckpt = weight_files[-1]
        model.load_weights(latest_ckpt)
        model.build(tf.TensorShape([1, None])) 

        input_indices = [char2idx[c] for c in seed]

        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

        generated = list(seed)
        most_likely_chars = []
        activations = []

        for i in range(len(input_indices)):

            input_tensor = tf.expand_dims([input_indices[i]], axis=0)

            predictions = model(input_tensor)
            logits = predictions[:, -1, :] / temperature  

            intermediate_model = tf.keras.Model(
                inputs=model.input,
                outputs=model.get_layer('rnn_layer').output
            )
            rnn_output = intermediate_model(input_tensor) 

            neuron_index = 942
            neuron_activation = rnn_output[0, -1, neuron_index].numpy()
            activations.append(neuron_activation)

            probs = tf.nn.softmax(logits, axis=-1)
            top_probs, top_indices = tf.math.top_k(probs, k=top_n)
            top_chars = [idx2char[idx.numpy()] for idx in top_indices]
            most_likely_chars.append(list(zip(top_chars, top_probs.numpy())))

        return ''.join(generated), generated, most_likely_chars, activations
    

    def plot_loss(self, history):
        plt.figure(figsize=(8, 5))
        train_loss = history.history['loss']
        val_loss = history.history.get('val_loss', [])
        
        epochs = range(1, len(train_loss) + 1) 

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label='Training Loss')
    
        if val_loss:
            plt.plot(epochs, val_loss, label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
