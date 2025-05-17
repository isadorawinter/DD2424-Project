from DataHandler import DataHandler
from RNN import RNN
from LSTM import LSTM
from ModelUtils import ModelUtils
import os
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
    
    lstm = LSTM()
    model = lstm.build_model(batch_size=batch_size,
                    sequence_length=sequence_length,
                    vocab_size=dataHandler.vocab_size,
                    embedding_dim=embedding_size,
                    n_rnn_units=n_rnn_units,
                    learning_rate=learning_rate,
                    num_layers=2)
    modelUtils = ModelUtils()
    modelUtils.train_model(model, dataHandler.dataset_batches, 10, 'RNNcheckpoints')
    rebuilt_model = lstm.build_model(
            batch_size=1,
            sequence_length=1,
            vocab_size=dataHandler.vocab_size,
            embedding_dim=embedding_size,
            n_rnn_units=n_rnn_units,
            learning_rate=learning_rate
        )
    stri = modelUtils.generate_text(
        rebuilt_model,
        checkpoint_dir=os.path.join('checkpoints', 'RNNcheckpoints'),
        char2idx=dataHandler.char2index,
        idx2char=dataHandler.index2char,
        seed="ROMEO: ",
        gen_length=20,
        temperature=0.8
    )
    print(stri)