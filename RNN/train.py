from DataHandler import DataHandler
from RNN import RNN
from LSTM import LSTM
from GRU import GRU
from ModelUtils import ModelUtils
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


if __name__ == '__main__':
    sequence_length = 100
    batch_size = 64
    embedding_size=256
    n_rnn_units = 1024
    learning_rate= 1e-3
    dataHandler = DataHandler(
        dataset_file_name='shakespeare.txt', 
        dataset_file_origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt',
        sequence_length=sequence_length, 
        batch_size=batch_size)
    
    #lstm = LSTM()
    #gru = GRU()
    rnn = RNN()
    #model = lstm.build_model(batch_size=batch_size,
    #model = gru.build_model(batch_size=batch_size,
    model = rnn.build_model(batch_size=batch_size,
                    sequence_length=sequence_length,
                    vocab_size=dataHandler.vocab_size,
                    embedding_dim=256,
                    n_rnn_units=1024,
                    learning_rate=learning_rate)#,
                    #num_layers=1)
                    #num_layers=2)

                    
    modelUtils = ModelUtils()
    modelUtils.train_model(model, dataHandler.dataset_batches, 10, 'RNNcheckpoints')  
    #rebuilt_model = lstm.build_model(
    #rebuilt_model = gru.build_model(
    rebuilt_model = rnn.build_model(
            batch_size=64,
            sequence_length=1,
            vocab_size=dataHandler.vocab_size,
            embedding_dim=256,
            n_rnn_units=1024,
            learning_rate=learning_rate#,
            #num_layers=1
            #num_layers=2
        )
    stri = modelUtils.generate_text(
        rebuilt_model,
        checkpoint_dir=os.path.join('checkpoints', 'RNNcheckpoints'),
        char2idx=dataHandler.char2index,
        idx2char=dataHandler.index2char,
        seed="ROMEO: ",
        gen_length=20,
        temperature=0.8,
        top_n=5
    )
    stri, generated, most_likely_chars, activations = modelUtils.visualizing_predictions(
        rebuilt_model,
        checkpoint_dir=os.path.join('checkpoints', 'RNNcheckpoints'),
        char2idx=dataHandler.char2index,
        idx2char=dataHandler.index2char,
        seed="ROMEO: ",
        gen_length=20,
        temperature=0.8,
        top_n=5
    )

    def visualize_predictions(generated, most_likely_chars, activations):
        
        steps = len(generated)
        rows = 6 

        fig, ax = plt.subplots(figsize=(steps * 0.6, rows * 0.8))
        ax.set_xlim(0, steps)
        ax.set_ylim(0, rows)
        ax.axis('off')

        def get_prob_color(prob):
            if prob >= 0.7:
                return "#ff0000"  
            elif prob >= 0.3:
                return "#ff9999" 
            else:
                return "#ffffff" 
            
        def get_activation_color(activation):
            if abs(activation) == 0:
                return "#ffffff"   
            elif activation >= 0.5:
                return '#39ff14'  
            elif activation > 0.0:
                return '#ccffcc'  
            elif activation <= -0.5:
                return "#242493"  
            elif activation < 0.0:
                return "#a0c9d7"  
            else:
                return "#ffffff"  

        for col, (char, activation) in enumerate(zip(generated, activations)):
            color = get_activation_color(activation)
            ax.add_patch(patches.Rectangle((col, 5), 1, 1, edgecolor='black', facecolor=color))
            ax.text(col + 0.5, 5.5, char, ha='center', va='center', fontsize=12, fontweight='bold')

        for col, step in enumerate(most_likely_chars):
            chars, probs = step[0]

            for row in range(5):
                char = chars[row]
                prob = probs[row]
                color = get_prob_color(prob)
                char_to_display = '\\n' if char == '\n' else char
                ax.add_patch(patches.Rectangle((col, 4 - row), 1, 1, edgecolor='black', facecolor=color))
                ax.text(col + 0.5, 4 - row + 0.5, char_to_display, ha='center', va='center', fontsize=10)

        plt.tight_layout()
        plt.show()

  
    print("predicted output", stri)
    print("generated", generated)
    print("most_likely_chars", most_likely_chars)
    print("activations", activations)

    visualize_predictions(generated, most_likely_chars, activations)
