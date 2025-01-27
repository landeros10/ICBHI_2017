"""
data.py
Author: landeros10
Date: 2025-01-25
Description: Contains data loading and preprocessing functions.
"""
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
import numpy as np
import os
import zipfile
import glob
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter
import zipfile

from src.logger import logger

SAMPLE_RATE = 16000
LOW_FREQ = 150
HIGH_FREQ = 800
ANNOTATIONS = ["start_time", "end_time", "crackles", "wheezes"]
GLOBAL_MEAN = 3.1238432632475526e-07
GLOBAL_STD = 0.03743929027120682

def parse_example(file_path):
    """
    Parse a single audio file and its associated annotation file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict: Parsed metadata and annotations.
    """

    annotation_file = file_path.replace(".wav", ".txt")
    annotation_df = pd.read_csv(annotation_file, sep="\t", names=ANNOTATIONS)

    filename = os.path.basename(file_path)
    parts = filename.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected filename format: {filename}")
    
    return {
        "patient_id": int(parts[0]),
        "recording_index": parts[1],
        "chest_location": parts[2],
        "acquisition_mode": parts[3],
        "equipment_type": parts[4].split(".")[0],
        "file_path": file_path,
        "annotations": annotation_df,
    }


def load_data(data_dir):
    """
    Load and parse audio files and their annotations from the specified directory.
    
    Args:
        data_dir (str): Directory containing audio files and annotations.
        
    Returns:
        pd.DataFrame: DataFrame containing parsed metadata and annotations.
    """
    logger.info(f"Loading data from {data_dir}")
    paths = glob.glob(os.path.join(data_dir, '*.wav'))
    audio_files = [parse_example(filename) for filename in paths]

    data_df = pd.DataFrame(audio_files)
    logger.info(f"Loaded {len(data_df)} audio files with variables: {data_df.columns.tolist()}")
    return data_df


def split_data(df, val_split, random_state):
    """
    Split dataframe into training and validation set.
    
    Args:
        df (pd.DataFrame): dataframe containing row examples
        val_split (float): Float value from 0.0 to 1.0 indicating validation proportion
    
    Returns:
        tuple: training and validation pandas dataframe
    """
    logger.info(f"Splitting data with validation split: {val_split}")
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=random_state)
    return train_df, val_df
    

def load_dataset(df, config, train_mode=True, transform=None):
    """
    Create a pytorch Dataset from pandas dataframe
    
    Args:
        df (pd.DataFrame): DataFrame containing file path, metadata, and annotations.
        config (dict): Configuration dictionary containing hyperparameters.
        train_mode (bool): Whether to use training mode or not.
    """
    logger.info(f"Creating a dataset with {len(df)} samples; Train mode: {train_mode}")
    dataset = RespiratoryDataset(
        df,
        clip_length=config["clip_length"],
        transform=transform,
        train_mode=train_mode,
    )
    return dataset


def split_test_ICBHI():
    labels_file = "./data/sample_labels.txt"
    test_dir = "./data/test"
    train_dir = "./data/train"
    data_dir = "./data"

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    with open(labels_file, "r") as f:
        for line in f:
            basename, subset = line.strip().split()
            src_audio_path = os.path.join(data_dir, f"{basename}.wav")
            src_annotation_path = os.path.join(data_dir, f"{basename}.txt")
            if subset == "test":
                dst_audio_path = os.path.join(test_dir, f"{basename}.wav")
                dst_annotation_path = os.path.join(test_dir, f"{basename}.txt")
            else:
                dst_audio_path = os.path.join(train_dir, f"{basename}.wav")
                dst_annotation_path = os.path.join(train_dir, f"{basename}.txt")
            
            if os.path.exists(src_audio_path):
                os.rename(src_audio_path, dst_audio_path)
            
            if os.path.exists(src_annotation_path):
                print(src_annotation_path)
                os.rename(src_annotation_path, dst_annotation_path)


def download_ICBHI():
    url = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"
    save_path = "./data/ICBHI_final_database.zip"
    extract_path = "./data"

    # Ensure the data directory exists
    os.makedirs(extract_path, exist_ok=True)

    # Skip download if data already exists
    if any(os.scandir(extract_path)):
        logger.info(f"Dataset already exists at {extract_path}. Skipping download.")
        return
    
    logger.info("Downloading dataset...")
    os.system(f"wget --no-check-certificate -O {save_path} {url}")

    logger.info("Extracting dataset...")
    with zipfile.ZipFile(save_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            member_filename = os.path.basename(member)
            if member_filename:  # This avoids directories
                target_path = os.path.join(extract_path, member_filename)
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

    logger.info("Extraction complete. Cleaning up...")
    os.remove(save_path)
    logger.info("Dataset is ready to use!")


def clip_audio(waveform, sr, clip_length, random_start=True):
    """
    Clip the audio waveform to the specified length in secods.

    Args:
        waveform (torch.Tensor): Audio waveform.
        sr (int): Sampling rate of the waveform.
        clip_length (float): Desired length of the audio clip in seconds.
        random_start (bool, optional): Whether to randomly select the start position. Defaults to True.
    
    Returns:
        tuple: Clipped waveform and start time of the clip.
    """
    start_time = 0.0
    if clip_length is None:
        return waveform, start_time
    
    target_length = int(clip_length * sr)
    waveform_length = waveform.shape[1]

    if waveform_length > target_length:
        if random_start:
            start = np.random.randint(0, waveform_length - target_length)
        else:
            start = (waveform_length - target_length) // 2
        start_time = start / sr
        waveform = waveform[:, start:start + target_length]
    elif waveform_length < target_length:
        padding = target_length - waveform_length
        waveform = torch.nn.functional.pad(waveform, (0, padding), mode="constant", value=0)
    return waveform, start_time


def get_clip_labels(clip_length, annotations_df, start_time):
    """
    Determine if the audio clip overlaps with respiratory cycles that contain crackles or wheezes.

    Args:
        annotations_df (pd.DataFrame): DataFrame containing respiratory cycle annotations.
        start_time (float): Start time of the audio clip in seconds.

    Returns:
        list: [crackles, wheezes] indicating presence (1) or absence (0) of respiratory conditions.
    """
    if clip_length is None:
        return list(annotations_df[["crackles", "wheezes"]].max(axis=0))
    end_time = start_time + clip_length
    overlaps = annotations_df[
        (annotations_df["start_time"] < end_time) & (annotations_df["end_time"] > start_time)
    ]

    if len(overlaps) > 0:
        return list(overlaps[["crackles", "wheezes"]].max(axis=0))
    
    return [0, 0]


def normalize_waveform(waveform, mean=GLOBAL_MEAN, std=GLOBAL_STD):
    """
    Normalize the audio waveform to have zero mean and unit variance.
    Waveform should have shape (channels, time_steps).
    
    Args:
        waveform (torch.Tensor): Audio waveform.
    
    Returns:
        torch.Tensor: Normalized waveform
    """

    if mean is None:
        mean = waveform.mean()
    if std is None:
        std = waveform.std()
    waveform = (waveform - mean) / (std + 1e-8)

    return waveform


def resample_waveform(waveform, original_sr, target_sr):
    """
    Resample the audio waveform to the target sampling rate.
    
    Args:
        waveform (torch.Tensor): Audio waveform.
        original_sr (int): Original sampling rate of the waveform.
        target_sr (int): Target sampling rate.
    
    Returns:
        torch.Tensor: Resampled waveform.
    """
    if original_sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = transform(waveform)
    return waveform


def denoise_waveform(waveform, sampling_rate, low_freq, high_freq):
    """
    Band pass filter to remove noise from the audio waveform.
    
    Args:
        waveform (torch.Tensor): Audio waveform.
        low_freq (int, optional): Low frequency cutoff. Defaults to None.
        high_freq (int, optional): High frequency cutoff. Defaults to None.

    Returns:
        torch.Tensor: Denoised waveform
    """
    waveform = torchaudio.functional.lowpass_biquad(waveform, sampling_rate, high_freq)
    waveform = torchaudio.functional.highpass_biquad(waveform, sampling_rate, low_freq)
    return waveform
    

def mono_waveform(waveform):
    if waveform.shape[0] > 1:
        print("Converting stereo to mono. wavform shape: {waveform.shape}")
        waveform = torchaudio.functional.downmix_mono(waveform)
    return waveform


def preprocess_audio(waveform, original_sr, target_sr, low_freq=None, high_freq=None):
    """
    Preprocess audio waveform:
    (1) Denoise the waveform with a bandpass filter.
    (2) Resample the waveform to the target sampling rate.
    (3) Convert the waveform to mono.

    Args:
        waveform (torch.Tensor): The input waveform.

    Returns:
        torch.Tensor: The processed input tensor.
    """
    # Denoise waveform
    if low_freq is not None and high_freq is not None:
        waveform = denoise_waveform(waveform, original_sr, low_freq, high_freq)

    # Resample waveform
    waveform = resample_waveform(waveform, original_sr, target_sr)

    # Convert to mono
    waveform = mono_waveform(waveform)

    return waveform


class RespiratoryDataset(Dataset):
    def __init__(self, df, sampling_rate=16000, low_freq=150, high_freq=800, clip_length=None, transform=None, train_mode=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file path, metadata, and annotations.
            sampling_rate (int, optional): Desired sampling rate. Defaults to 16000.
            clip_length (float, optional): Length of the audio clip in seconds. Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = df
        self.sampling_rate = sampling_rate
        self.clip_length = clip_length
        self.transform = transform
        self.train_mode = train_mode
        self.low_freq = low_freq
        self.high_freq = high_freq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["file_path"]
        if not os.path.exists(audio_path):
            logger.warning(f"Missing file: {audio_path}")
            return None

        # Load audio file, resample if necessary, and clip
        waveform, sr = torchaudio.load(audio_path)
        original_length = waveform.shape[1] / sr

        # Preprocess audio
        waveform = preprocess_audio(waveform,
                                    original_sr=sr,
                                    target_sr=self.sampling_rate,
                                    low_freq=self.low_freq,
                                    high_freq=self.high_freq,)

        # Clip waveform
        waveform, start_time = clip_audio(waveform, self.sampling_rate, self.clip_length, random_start=self.train_mode)

        # Get annotations for the current clip
        labels = get_clip_labels(self.clip_length, row["annotations"], start_time)

        # Apply any additional transforms
        if self.transform:
            waveform = self.transform(waveform)

        sample = {
            "x": waveform.to(dtype=torch.float32),
            "y": torch.tensor(labels, dtype=torch.float32),
            "metadata": {
                "file_path": audio_path,
                "patient_id": int(row["patient_id"]),
                "chest_location": str(row["chest_location"]),
                "equipment_type": str(row["equipment_type"]),
                "original_sr": int(sr),
                "original_length": float(original_length),
                "clip_start_time": float(start_time),
            }
        }

        return sample
