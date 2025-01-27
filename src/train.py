"""
train.py
Author: landeros10
Date: 2025-01-25
Description: Contains training and evaluation functions.
"""

import glob
import math
import random
import os
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import torchaudio
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

from src.data import download_ICBHI, load_data, split_data, load_dataset, preprocess_audio, SAMPLE_RATE
from src.models import load_model
from src.logger import logger


# W&B Sweep Configuration
sweep_config = {
    "method": "random",  # or could do  'bayes'
    "metric": {
        "name": "Val/Loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "values": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4]
        },
        "batch_size": {
            "values": [16, 24, 32, 40, 48, 56]
        },
        "snr": {
            "values": [10, 15, 20, 25, 30]
        },
    }
}

# Default configuration
default_config = {
    "data_dir": "./data/train",
    "checkpoint_dir": "./checkpoints",
    "model_type": "wav2vec",
    "clip_length": 5.0,
    "n_mels": 64,
    "hann_window": 0.064,
    "window_shift": 0.032,
    "num_classes": 2,
    "epochs": 50,
    "train_subsampling": 0,
    "batch_size": 8,
    "num_workers": 4,
    "learning_rate": 1e-6,
    "optimizer_decay": 0.01,
    "noise_p": 0.5,
    "noise_type": "hospital",
    "noise_dir": "./data/hospital_noise",
    "snr": 10,
    "val_split": 0.2,
    "random_seed": 42,
    "criterion": "BCEWithLogitsLoss",
    "resume_checkpoint": False,
    "project": "respiratory-sound-multilabel",
}


def parse_args():
    """
    Parses command-line arguments for the training script.
    
    Returns:
        dict: Dictionary containing configuration options such as data paths, model parameters, 
          training settings, and optimization details
    """
    parser = argparse.ArgumentParser(description="Train a respiratory sound classification model.")

    parser.add_argument("--data_dir", type=str, default=default_config["data_dir"], 
                        help="Path to the directory containing training data.")
    parser.add_argument("--checkpoint_dir", type=str, default=default_config["checkpoint_dir"], 
                        help="Path to save training checkpoints.")

    parser.add_argument("--model_type", type=str, default=default_config["model_type"], 
                        help="Type of model architecture to use (e.g., wav2vec).")
    parser.add_argument("--num_classes", type=int, default=default_config["num_classes"], 
                        help="Number of output classes for classification.")

    parser.add_argument("--clip_length", type=float, default=default_config["clip_length"], 
                        help="Duration of each audio clip in seconds.")
    parser.add_argument("--noise_p", type=float, default=default_config["noise_p"], 
                        help="Probability of applying noise augmentation to audio clips.")
    parser.add_argument("--noise_type", type=str, default=default_config["noise_type"], 
                        help="Type of noise to apply (e.g., 'hospital' or 'random').")
    parser.add_argument("--noise_dir", type=str, default=default_config["noise_dir"], 
                        help="Directory containing noise audio files for augmentation.")
    parser.add_argument("--snr", type=float, default=default_config["snr"], 
                        help="Signal-to-noise ratio (SNR) in decibels for noise augmentation.")

    parser.add_argument("--criterion", type=str, choices=["BCEWithLogitsLoss", "CrossEntropyLoss"], default="BCEWithLogitsLoss", 
                        help="Loss function to use for training.")
    parser.add_argument("--epochs", type=int, default=default_config["epochs"], 
                        help="Total number of training epochs.")
    parser.add_argument("--train_subsampling", type=int, default=default_config["train_subsampling"], 
                        help="Number of samples to use from the training dataset for subsampling (0 for full dataset).")
    parser.add_argument("--batch_size", type=int, default=default_config["batch_size"], 
                        help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=default_config["num_workers"], 
                        help="Number of worker threads for data loading.")
    parser.add_argument("--learning_rate", type=float, default=default_config["learning_rate"], 
                        help="Initial learning rate for the optimizer.")
    parser.add_argument("--optimizer_decay", type=float, default=default_config["optimizer_decay"], 
                        help="Weight decay factor for the optimizer to prevent overfitting.")

    parser.add_argument("--val_split", type=float, default=default_config["val_split"], 
                        help="Proportion of data to be used for validation (e.g., 0.2 for 20%%).")
    parser.add_argument("--random_seed", type=int, default=default_config["random_seed"], 
                        help="Random seed for reproducibility.")
    parser.add_argument("--resume_checkpoint", action="store_true", 
                        help="Resume training from the latest checkpoint if available.")
    parser.add_argument("--sweep", action="store_true", 
                        help="Enable W&B sweep for hyperparameter tuning.")
    parser.add_argument("--download_ICBHI", action="store_true", 
                        help="Download the ICBHI dataset to the data directory.")
    parser.add_argument("--project", type=str, default=default_config["project"], 
                        help="Name of the Weights & Biases project to log training runs.")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()
    config = vars(args)

    logger.info("[TRAIN] - Parsed Configuration")
    return config


def set_seed(seed):
    """ Sets the random seed for reproducibility. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_metrics(num_labels):
    """ Sets up the metrics for training and evaluation.
    
    Args:
        num_labels (int): Number of classes.
    
    Returns:
        torchmetrics.MetricCollection: Collection of metrics.
    """
    metrics = MetricCollection({
        "accuracy": Accuracy(task='multilabel', threshold=0.5, num_labels=num_labels, average='none'),
        "precision": Precision(task='multilabel', threshold=0.5, num_labels=num_labels, average='none'),
        "recall": Recall(task='multilabel', threshold=0.5, num_labels=num_labels, average='none')
    })
    return metrics


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers):
    """
    Creates DataLoader objects for the training and validation datasets.
    
    Args:
        train_dataset (RespiratoryDataset): The training dataset.
        val_dataset (RespiratoryDataset): The validation dataset.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
        num_workers (int, optional): Number of worker threads. Defaults to NUM_WORKERS.
    
    Returns:
        tuple: DataLoader objects for the training and validation datasets.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def setup_training(model, config, num_training_steps=None):
    """
    Sets up the training components including the loss function, optimizer, and learning rate scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        config (dict): Configuration dictionary containing hyperparameters.
        num_training_steps (int, optional): The total number of training steps. Defaults to None.

    Returns:
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
    """
    logger.info("[TRAIN] - Setting up training components...")

    # Set loss function
    if config["criterion"] == "BCEWithLogitsLoss":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config["criterion"] == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss function specified.")
    logger.debug(f"Criterion: {criterion}")

    # Optimizer and Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["optimizer_decay"])
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger.debug(f"Optimizer: {optimizer}")
    logger.debug(f"Learning Rate Scheduler: {lr_scheduler}")
    return criterion, optimizer, lr_scheduler


def log_training_params(config):
    """
    Logs training parameters with categorized groupings for better readability.
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters.
    """
    grouped_params = {
        "General Settings": {
            "Data Directory": config["data_dir"],
            "Checkpoint Directory": config["checkpoint_dir"],
            "Random Seed": config["random_seed"],
            "Resume from Checkpoint": config["resume_checkpoint"],
            "Project Name": config["project"],
            "Download ICBHI Dataset": config["download_ICBHI"],
            "Enable W&B Sweep": config["sweep"],
        },
        "Model Parameters": {
            "Model Type": config["model_type"],
            "Number of Classes": config["num_classes"],
            "Criterion": config["criterion"],
        },
        "Audio Processing": {
            "Clip Length (sec)": config["clip_length"],
            "Number of Mel Filters": config["n_mels"],
            "Hann Window Length (sec)": config["hann_window"],
            "Window Shift (sec)": config["window_shift"],
        },
        "Training Settings": {
            "Batch Size": config["batch_size"],
            "Number of Workers": config["num_workers"],
            "Epochs": config["epochs"],
            "Validation Split": config["val_split"],
            "Training Subsampling": config["train_subsampling"],
        },
        "Data Augmentation": {
            "Noise Type": config["noise_type"],
            "Noise Directory": config["noise_dir"],
            "Noise Probability": config["noise_p"],
            "SNR (dB)": config["snr"],
            "Time Stretch Probability": config["time_stretch_p"],
            "Time Stretch Min Factor": config["time_stretch_min"],
            "Time Stretch Max Factor": config["time_stretch_max"],
        },
        "Optimization": {
            "Learning Rate": config["learning_rate"],
            "Optimizer Weight Decay": config["optimizer_decay"],
        }
    }

    logger.info("[TRAIN] - Training Configuration")
    logger.info("=" * 50)

    for category, params in grouped_params.items():
        logger.info(f"[TRAIN] - {category}:")
        logger.info("-" * 50)
        for key, value in params.items():
            logger.info(f"[TRAIN] - {key:<30}: {value}")
        logger.info("-" * 50)

    logger.info("=" * 50)


def load_noise_tensor(config, length):
    """
    Loads a noise tensor from a file or generates random noise.
    if the noise_type is set to "hospital", the noise tensor is loaded from a file.
    else, random noise is generated.
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters.
        length (int): The length of the noise tensor.
    
    Returns:
        torch.Tensor: The noise tensor.
        str: The noise file path if loaded from file.
    """
    noise_type = config.get("noise_type", "random")
    noise = torch.randn(1, length)
    noise_file = ""
    if noise_type == "hospital":
        hospital_noise_dir = config["noise_dir"]
        
        if not os.path.exists(hospital_noise_dir):
            logger.warning("Hospital noise directory not found. Using random noise.")
            return noise, noise_file
        
        noise_files = glob.glob(os.path.join(hospital_noise_dir, "*.wav"))
        if not noise_files:
            logger.warning("No hospital noise files found. Using random noise.")
            return noise, noise_file
        
        noise_file = random.choice(noise_files)
        noise, noise_sr = torchaudio.load(noise_file)
        noise = preprocess_audio(noise, noise_sr, SAMPLE_RATE)
    return noise, noise_file


def add_noise(waveform, noise, snr_db):
    """
    Adds noise to a waveform with controlled scaling.

    Args:
        waveform (torch.Tensor): Clean audio waveform.
        noise (torch.Tensor): Noise waveform.
        snr_db (float): Desired signal-to-noise ratio in decibels.

    Returns:
        torch.Tensor: Noisy waveform.
    """
    if torch.isnan(noise).any():
        logger.warning("NaN detected in noise tensor.")
        return waveform

    signal_power = torch.mean(waveform ** 2)
    noise_power = torch.mean(noise ** 2)

    if noise_power == 0:
        logger.warning("Warning: Noise has zero power, returning original waveform.")
        return waveform

    snr_ratio = 10 ** (snr_db / 10)
    scaling_factor = torch.sqrt(signal_power / (snr_ratio * noise_power + 1e-8))
    noisy_waveform = waveform + scaling_factor * noise

    if torch.isnan(noisy_waveform).any():
        logger.warning("NaN detected in manually added noise.")
        return waveform

    return noisy_waveform


def build_augmentation_fn(config):
    """ 
    Build data augmentations for audio data.
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters.
    Returns:
        callable: Transform function that applies augmentations to the audio waveform.
    """
    logger.info("[TRAIN] - Building data augmentations...")

    def augment(waveform):
        # Apply noise
        p = config.get("noise_p", 0.5)
        if torch.rand(1).item() < p:
            noise, noise_file = load_noise_tensor(config, waveform.size(1)) # Raw noise tensor
            waveform = add_noise(waveform, noise, torch.tensor([config.get("snr", 10)]))

        # TODO: Apply time stretching
        # TODO: Apply pitch shifting
        
        # Squeeze to remove channel dimension
        waveform = torch.squeeze(waveform)
        return waveform
    return augment


def train_one_epoch(model, train_loader, optimizer, criterion, metrics, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        metrics (torchmetrics.MetricCollection): The metrics to be computed.
        device (torch.device): The device to run the training on.
        epoch (int): The current epoch number.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0

    metrics.to(device)
    metrics.reset()
    for i, batch in enumerate(train_loader):
        logger.debug(f"Processing batch {i + 1}/{len(train_loader)}")
        inputs = batch["x"].to(device)
        labels = batch["y"].to(device)

        outputs = model(inputs).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        metrics(torch.sigmoid(outputs) > 0.5, labels.int())

        logger.debug(f"Batch {i + 1} loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    metrics_results = metrics.compute()
    
    wandb.log({"Train/Loss": avg_loss, "epoch": epoch})
    for metric_name, metric_value in metrics_results.items():
        if metric_value.numel() > 1:
            for i, value in enumerate(metric_value):
                wandb.log({f"Train/Metrics/{metric_name}_{i}": value.item(), "epoch": epoch})
        else:
            wandb.log({f"Train/Metrics/{metric_name}": metric_value.item(), "epoch": epoch})            
    
    logger.info(f"[TRAIN] - Training Loss: {avg_loss:.4f}")
    logger.info(f"[TRAIN] - Training Metrics: {metrics_results}")
    return avg_loss


def evaluate(model, val_loader, criterion, metrics, device, epoch):
    """
    Evaluates the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): The loss function.
        metrics (torchmetrics.MetricCollection): The metrics to be computed.
        device (torch.device): The device to run the evaluation on.
        epoch (int): The current epoch number.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    val_loss = 0.0
    
    metrics.to(device)
    metrics.reset()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logger.debug(f"Processing validation batch {i + 1}/{len(val_loader)}")
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            metrics.update(torch.sigmoid(outputs) > 0.5, labels.int())
            logger.debug(f"Validation batch {i + 1} loss: {loss.item():.4f}")

    avg_val_loss = val_loss / len(val_loader)
    metrics_results = metrics.compute()

    wandb.log({"Val/Loss": avg_val_loss, "epoch": epoch})
    for metric_name, metric_value in metrics_results.items():
        if metric_value.numel() > 1:
            for i, value in enumerate(metric_value):
                wandb.log({f"Val/Metrics/{metric_name}_{i}": value.item(), "epoch": epoch})
        else:
            wandb.log({f"Val/Metrics/{metric_name}": metric_value.item(), "epoch": epoch})

    logger.info(f"[TRAIN] -Validation Loss: {avg_val_loss:.4f}")
    logger.info(f"[TRAIN] -Validation Metrics: {metrics_results}")
    return avg_val_loss


def save_checkpoint(model, val_loss, best_val_loss, epoch, config):
    """
    Saves the model checkpoint as a W&B artifact.
    
    Args:
        model (torch.nn.Module): The model to be saved.
        val_loss (float): The validation loss.
        best_val_loss (float): The best validation loss.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing hyperparameters.
    
    Returns:
        float: The best validation loss
    """    
    if val_loss < best_val_loss:
        checkpoint_dir = config["checkpoint_dir"]
        logger.info(f"[TRAIN] -Saving checkpoint for epoch {epoch} with validation loss {val_loss:.4f}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"latest_checkpoint.pth")
        torch.save(model.state_dict(), model_path)

        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch + 1}"])
        logger.info("[TRAIN] -Best model saved as W&B artifact")

        best_val_loss = val_loss
    return best_val_loss


def train(config):
    """
    Main training function.
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters.
    """

    logger.info("[TRAIN] -Starting Training...")
    set_seed(config["random_seed"])
    log_training_params(config)

    # Load Data
    df = load_data(config["data_dir"])
    if config["train_subsampling"]:
        n = min(config["train_subsampling"], len(df))
        df = df.sample(n=n, random_state=config["random_seed"])
    train_df, val_df = split_data(df, config["val_split"], config["random_seed"])
    train_dataset = load_dataset(train_df, config, train_mode=True, transform=build_augmentation_fn(config))
    val_dataset = load_dataset(val_df, config, train_mode=False, transform=torch.squeeze)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config["batch_size"], config["num_workers"])

    # Load Model and Optimization Parameters
    model, processor = load_model(model_type=config["model_type"], num_labels=config["num_classes"])
    criterion, optimizer, lr_scheduler = setup_training(
        model,
        config,
        num_training_steps=config["epochs"] * len(train_loader),
    )
    metrics = setup_metrics(config["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Training on device: {device}")

    model.to(device)
    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        logger.info(f"[TRAIN] -Starting epoch {epoch + 1}/{config['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, metrics, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, metrics, device, epoch)
        best_val_loss = save_checkpoint(model, val_loss, best_val_loss, epoch, config)

    logger.info("[TRAIN] -Training Completed")


def run_sweep():
    """ W&B Sweep callback function. """
    wandb.init(name=get_run_name())
    sweep_config = dict(wandb.config)  # Convert wandb.config to dictionary
    config = {**default_config, **sweep_config}    
    train(config)
    wandb.finish()


def get_run_name():
    """ Generates a unique run name based on the current timestamp. """
    return f"GC-{datetime.now().strftime('%Y%m%d-%H%M')}"


if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info(f"[TRAIN] - Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("[TRAIN] - Using CPU for training.")

    config = parse_args()
    if config["debug"]:
        logger.setLevel("DEBUG")

    if config["download_ICBHI"]:
        logger.info("[TRAIN] - Downloading ICBHI dataset...")
        download_ICBHI()

    if config["sweep"]:
        logger.info("[TRAIN] - Starting W&B Sweep...")
        sweep_id = wandb.sweep(sweep_config, project=config["project"])
        wandb.agent(sweep_id, function=run_sweep, count=15)
    else:
        wandb.init(
            project=config["project"],
            config=config,
            name=get_run_name(),
        )
        train(config)

