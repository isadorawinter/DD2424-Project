import tensorflow as tf
import pathlib
import numpy as np
class DataHandler:
    def __init__(self, dataset_file_name, dataset_file_origin, sequence_length, batch_size, train_end, val_end):
        self.dataset_file_name = dataset_file_name
        self.dataset_file_origin = dataset_file_origin
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_end = train_end
        self.val_end = val_end

        self.text = self.read_data()
        self.char2index, self.index2char, self.text_as_int, self.vocab_size = self.process_data()
        self.split_data(self.train_end, self.val_end)
        self.train_batches = self.create_batches(self.train_data)
        self.val_batches = self.create_batches(self.val_data)
        self.test_batches = self.create_batches(self.test_data)

    def read_data(self):
        '''
        Downloads and reads the contents of a text dataset.

        Args:
            dataset_file_name (str): The name to save the dataset file as.
            dataset_file_origin (str): The URL where the dataset can be downloaded.

        Returns: 
            str: Entire content of the file as a single string.
        '''
        cache_dir = '.tmp'
        
        dataset_file_path = tf.keras.utils.get_file(
            fname=self.dataset_file_name,
            origin=self.dataset_file_origin,
            cache_dir=pathlib.Path(cache_dir).absolute()
        )

        text = open(dataset_file_path, mode='r').read()
        return text
    

    def process_data(self):
        '''
        Processes raw text into numeric representations for model input.

        Returns:
            tuple: A tuple containing:
                - char2index (dict): Mapping from characters to unique integer index.
                - index2char (np.ndarray): Mapping from indices to chars.
                - text_as_int (np.ndarray): Array of integer indices representing the text.
        '''
        vocab = sorted(list(set(self.text)))
        char2index = {char: index for index, char in enumerate(vocab)}
        index2char = np.array(vocab)
        text_as_int = np.array([char2index[char] for char in self.text])

        return char2index, index2char, text_as_int, len(vocab)
    
    def split_data(self, train_end, val_end):
        total_len = len(self.text_as_int)
        train_end = int(train_end * total_len)
        val_end = int(val_end * total_len)

        self.train_data = self.text_as_int[: train_end]
        self.val_data = self.text_as_int[train_end: val_end]
        self.test_data = self.text_as_int[val_end:]
    

    def create_batches(self, data):
        '''
        Creates batches of training sequences from encoded text for input.
        '''
        char_dataset = tf.data.Dataset.from_tensor_slices(data)
        sequences = char_dataset.batch(self.sequence_length+1, drop_remainder=True)
        dataset_sequences = sequences.map(self.split_input_target)
        return dataset_sequences.shuffle(10000).batch(self.batch_size, drop_remainder=True)

    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    
