from DataHandler import DataHandler
from RNN import RNN
from LSTM import LSTM
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


def phase_one():
    batch_size_choices = [32, 64, 128]
    n_trials = 10
    '''
    results_file = "RNN_results.csv"
    first_write = not os.path.exists(results_file)
    for batch_size in batch_size_choices:
        for trial in range(n_trials):
            seed = 42+trial
            set_global_seed(seed)
            learning_rate = log_uniform()

            ckpt_subdir = f"RNNcheckpoints_seed{seed}_bs{batch_size}_lr{learning_rate:.2e}"
            config = {
                "model_type": "rnn",
                "dataset_file_name": "shakespeare.txt",
                "dataset_file_origin": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
                "sequence_length": 100,
                "batch_size": batch_size,
                "train_end": 0.8,
                "val_end": 0.9,
                "embedding_size": 256,
                "n_rnn_units": 256,
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

    results_file = "LSTM1_results.csv"
    first_write = not os.path.exists(results_file)
    for batch_size in batch_size_choices:
        for trial in range(n_trials):
            seed = 42+trial
            set_global_seed(seed)
            learning_rate = log_uniform()

            ckpt_subdir = f"LSTM1checkpoints_seed{seed}_bs{batch_size}_lr{learning_rate:.2e}"
            config = {
                "model_type": "lstm",
                "dataset_file_name": "shakespeare.txt",
                "dataset_file_origin": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
                "sequence_length": 100,
                "batch_size": batch_size,
                "train_end": 0.8,
                "val_end": 0.9,
                "embedding_size": 256,
                "n_rnn_units": 256,
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
    '''
    results_file = "LSTM2_results.csv"
    first_write = not os.path.exists(results_file)
    for batch_size in batch_size_choices:
        for trial in range(n_trials):
            seed = 42+trial
            set_global_seed(seed)
            learning_rate = log_uniform()

            ckpt_subdir = f"LSTM2checkpoints_seed{seed}_bs{batch_size}_lr{learning_rate:.2e}"
            config = {
                "model_type": "lstm",
                "dataset_file_name": "shakespeare.txt",
                "dataset_file_origin": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
                "sequence_length": 100,
                "batch_size": batch_size,
                "train_end": 0.8,
                "val_end": 0.9,
                "embedding_size": 256,
                "n_rnn_units": 256,
                "learning_rate": learning_rate,
                "epochs": 20,
                "num_layers": 2,
                "checkpoint_subdir": ckpt_subdir
            }
            result = train_phase_one(config)
            with open(results_file, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=result.keys())
                    if first_write:
                        w.writeheader()
                        first_write = False
                    w.writerow(result)

def extend_training(config, ckpt_dir, extra_epochs):
    tf.keras.backend.clear_session()
    data_handler = DataHandler(
        dataset_file_name=config["dataset_file_name"],
        dataset_file_origin =config["dataset_file_origin"],
        sequence_length=config["sequence_length"],
        batch_size= config["batch_size"],
        train_end=config['train_end'],
        val_end=config['val_end']
    )

    # 2. Build model
    model_class = RNN if config["model_type"] == "rnn" else LSTM
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
    model.build((config["batch_size"], config["sequence_length"]))
    weights = os.path.join("checkpoints/" + ckpt_dir, "best.weights.h5")
    print(weights)
    model.load_weights(weights, skip_mismatch=True)

    mu = ModelUtils()
    history = mu.train_model(model,
                   train_data      = data_handler.train_batches,
                   val_data        = data_handler.val_batches,
                   epochs          = extra_epochs,   
                   checkpoint_subdir= ckpt_dir + "v2")
    best_val = min(history.history["val_loss"])
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    print(config["learning_rate"])
    print(config["batch_size"])
    print(best_val)
    print(best_epoch)

def plot_phase_two():
    csv_files = glob.glob("*_results.csv")
    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        arch = os.path.basename(path).split("_")[0].upper()   # infer architecture
        df["model"] = arch
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=(8, 6))

    for (arch, bs), grp in full.groupby(["model", "batch_size"]):
        grp = grp.sort_values("learning_rate")
        plotlabel = "RNN"
        if arch=="LSTM1":
            plotlabel = "1-LSTM"
        elif arch== "LSTM2":
            plotlabel = "2-LSTM"
            print(grp["val_loss"])
        
        plt.plot(
            grp["learning_rate"],
            grp["val_loss"],
            marker="o",
            label=f"{plotlabel} | Batch size: {bs}"
        )
    plt.ylim(bottom=min(full["val_loss"]) - 0.1)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Validation loss")
    plt.grid(True, linestyle="-", linewidth=0.5)
    plt.legend(fontsize=8)
    plt.savefig("Phase2.png")

def hidden_units():
    architectures = [
         {"model_type": "rnn", "batch_size": 64,"learning_rate": 0.0031954089406218927,
          "num_layers": 1},
         {"model_type": "lstm", "batch_size": 32, "learning_rate": 0.0031954089406218927, 
          "num_layers": 1},
         {"model_type": "lstm", "batch_size": 32, "learning_rate": 0.001064619082992022,
            "num_layers": 2}
    ]

    hidden_unit_sizes = [128, 256, 512, 1024]

    for arch_cfg in architectures:
        results_file = f"{arch_cfg['model_type']}_units_results.csv"
        first_write  = not os.path.exists(results_file)

        for hu in hidden_unit_sizes:
            seed = 42 + hu                          # different seed per width
            set_global_seed(seed)

            ckpt_subdir = (f"{arch_cfg['model_type']}_L{arch_cfg['num_layers']}"
                           f"_BS{arch_cfg['batch_size']}_HU{hu}")

            config = {
                "model_type"       : arch_cfg["model_type"],
                "dataset_file_name": "shakespeare.txt",
                "dataset_file_origin": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
                "sequence_length"  : 100,
                "batch_size"       : arch_cfg["batch_size"],
                "train_end"        : 0.8,
                "val_end"          : 0.9,
                "embedding_size"   : 256,
                "n_rnn_units"      : hu,           
                "learning_rate"    : arch_cfg["learning_rate"],
                "epochs"           : 1000,            
                "num_layers"       : arch_cfg["num_layers"],
                "checkpoint_subdir": ckpt_subdir,
                "seed"             : seed
            }

            result = train_phase_one(config)        

            with open(results_file, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=result.keys())
                if first_write:
                    w.writeheader(); first_write = False
                w.writerow(result)

             
def train_phase_one(config):
    tf.keras.backend.clear_session()
    t0 = time.time()
    # 1. Prepare data
    data_handler = DataHandler(
        dataset_file_name=config["dataset_file_name"],
        dataset_file_origin =config["dataset_file_origin"],
        sequence_length=config["sequence_length"],
        batch_size= config["batch_size"],
        train_end=config['train_end'],
        val_end=config['val_end']
    )

    # 2. Build model
    model_class = RNN if config["model_type"] == "rnn" else LSTM
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

    # 3. Train model
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
    
    
         

def train_and_generate(config):
    '''
    # 4. Rebuild model for generation
    gen_model = model_builder.build_model(
        batch_size=1,
        sequence_length=1,
        vocab_size=data_handler.vocab_size,
        embedding_dim=config["embedding_size"],
        n_rnn_units=config['n_rnn_units'],
        learning_rate=config['learning_rate'],
        num_layers=config['num_layers']
    )

    # 5. Generate text
    generated = model_utils.generate_text(
        gen_model,
        checkpoint_dir=os.path.join('checkpoints', config["checkpoint_subdir"]),
        char2idx=data_handler.char2index,
        idx2char=data_handler.index2char,
        seed=config["seed"],
        gen_length=config["gen_length"],
        temperature=config["temperature"],
        nucleus_sampling=config["nucleus_sampling"],
        p=config["p"]
    )
    print(generated)
    model_utils.plot_loss(history)

    config = {
        "model_type": "rnn",  # "lstm" or "rnn"
        "dataset_file_name": "shakespeare.txt",
        "dataset_file_origin": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
        "sequence_length": 100,
        "batch_size": 64,
        "train_end": 0.8,
        "val_end": 0.9,
        "embedding_size": 256,
        "n_rnn_units": 1024,
        "learning_rate": 1e-3,
        "epochs": 40,
        "num_layers": 1,
        "checkpoint_subdir": "RNNcheckpoints",
        "seed": "ROMEO: ",
        "gen_length": 200,
        "temperature": 0.8,
        "nucleus_sampling": True,
        "p": 0.9
    }
    '''

