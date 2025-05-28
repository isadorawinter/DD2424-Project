from DataHandler import DataHandler
from RNN import RNN
from LSTM import LSTM
from GRU import GRU
from ModelUtils import ModelUtils
import os
import numpy as np
import random
import tensorflow as tf
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import glob

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def log_uniform(low=1e-5, high=1e-2):
    return 10 ** np.random.uniform(np.log10(low), np.log10(high))

num_layers = 3

def phase_one():
    batch_size_choices = [32, 64, 128]
    n_trials = 10
   
    results_file = "GRU1_results.csv"
    first_write = not os.path.exists(results_file)
    
    for batch_size in batch_size_choices:
        
        for trial in range(n_trials):
            
            seed = 42+trial
            set_global_seed(seed)
            
            learning_rate = log_uniform()

            ckpt_subdir = f"GRUcheckpoints_seed{seed}_bs{batch_size}_lr{learning_rate:.2e}"
            config = {
                "model_type": "gru",
                "dataset_file_name": "shakespeare.txt",
                "dataset_file_origin": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
                "sequence_length": 100,
                "batch_size": batch_size,
                "train_end": 0.8,
                "val_end": 0.9,
                "embedding_size": 512,
                "n_rnn_units": 512,
                "learning_rate": learning_rate,
                "epochs": 20,
                "num_layers": 1,
                "checkpoint_subdir": ckpt_subdir
            }
            result = train_phase_one(config)
            with open(results_file, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=result.keys())
                    if first_write:
                        w.writeheader()
                        first_write = False
                    w.writerow(result)

             
def train_phase_one(config):
    tf.keras.backend.clear_session()
    t0 = time.time()
   
    data_handler = DataHandler(
        dataset_file_name=config["dataset_file_name"],
        dataset_file_origin =config["dataset_file_origin"],
        sequence_length=config["sequence_length"],
        batch_size= config["batch_size"],
        train_end=config['train_end'],
        val_end=config['val_end']
    )

    model_class = GRU              
    model_builder = model_class()
    model = model_builder.build_model(
        batch_size=config["batch_size"],
        sequence_length=config["sequence_length"],
        vocab_size=data_handler.vocab_size,
        embedding_dim=config["embedding_size"],
        n_rnn_units=config["n_rnn_units"],
        learning_rate=config["learning_rate"],
        num_layers=config["num_layers"]
    )

    model_utils = ModelUtils()
    history = model_utils.train_model(
        model,
        train_data=data_handler.train_batches,
        epochs=config['epochs'],
        checkpoint_subdir=config['checkpoint_subdir'],
        val_data=data_handler.val_batches
    )

    best_val = min(history.history["val_loss"])
    best_epoch = np.argmin(history.history["val_loss"]) + 1

    runtime = time.time() - t0

    return {
        "model":config["model_type"],
        "layers":config["num_layers"],
        "batch_size":config["batch_size"],
        "learning_rate":config["learning_rate"],
        "best_epoch": best_epoch,
        "val_loss": best_val,
        "runtime_s":round(runtime,1)
    }

if __name__ == '__main__':
    phase_one()
