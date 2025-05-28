import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from DataHandler import DataHandler
import os
import matplotlib.pyplot as plt

class ModelUtils:
    def create_checkpoint_callback(self, checkpoint_dir):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{checkpoint_dir}/best.weights.h5',
            save_best_only=True,
            save_weights_only=True
        )
    
    def train_model(self, model, train_data, val_data, epochs, checkpoint_subdir):
        full_path = os.path.join('checkpoints', checkpoint_subdir)
        os.makedirs(full_path, exist_ok=True)
        checkpoint_callback = self.create_checkpoint_callback(full_path)
        training_history = model.fit(train_data, 
                                     epochs=epochs,
                                     callbacks=[checkpoint_callback], 
                                     validation_data=val_data)
        return training_history
    
    def generate_text(self,
        model,
        checkpoint_dir,
        char2idx,
        idx2char,
        seed,
        gen_length,
        temperature,
        nucleus_sampling,
        p
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

            if(nucleus_sampling):
                next_id = self.nucleus_sample(logits[0], p).numpy()
            else:
                next_id = tf.random.categorical(logits, num_samples=1)[0, 0].numpy()

            generated.append(idx2char[next_id])
            input_tensor = tf.expand_dims([next_id], axis=0)

        return ''.join(generated)
    
    
    def nucleus_sample(self, logits, p=0.9):
        '''
        Implements nucleus (top-p) sampling, which samples only from the most
        probable subset whose combined probability is >=p.
        '''
        probs = tf.nn.softmax(logits)
        
        sorted_probs, sorted_indices = tf.sort(probs, direction='DESCENDING'), tf.argsort(probs, direction='DESCENDING')

        cumulative_probs = tf.math.cumsum(sorted_probs)

        nucleus_mask = cumulative_probs <= p

        nucleus_mask = tf.tensor_scatter_nd_update(
            nucleus_mask,
            indices=[[0]],
            updates=[True]
        )

        nucleus_probs = tf.where(nucleus_mask, sorted_probs, tf.zeros_like(sorted_probs))
        nucleus_probs /= tf.reduce_sum(nucleus_probs)

        sampled_idx = tf.random.categorical(tf.math.log([nucleus_probs]), num_samples=1)[0, 0]
        return sorted_indices[sampled_idx]
    
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
