"""
inference.py
Author: landeros10
Date: 2025-01-25
Description: Application for performing inference on respiratory sound classification.
"""

import sys
import argparse
import os
import json
import logging
import numpy as np
from time import time
from matplotlib import pyplot as plt
import torch
import torchaudio
import wandb
from sklearn.metrics import confusion_matrix
from scipy.ndimage import zoom
import pandas as pd

from src.models import load_model
from src.train import setup_metrics
from src.data import preprocess_audio, SAMPLE_RATE, ANNOTATIONS
from src.logger import logger

WANDB_PROJECT = "respiratory-sound-multilabel"
DEFAULT_MODEL_TYPE = "wav2vec"


def calculate_f1(results):
    """
    Calculates the F1 score for each class based on precision and recall.
    
    Args:
        results (dict or str): A dictionary containing metrics results or a path to the JSON file containing metric results.
    
    Returns:
        list: A list of F1 scores for each class.
    """
    if isinstance(results, str):
        with open(results, 'r') as f:
            results = json.load(f)

    precision = results['precision']
    recall = results['recall']

    f1_scores = [
        (2 * p * r) / (p + r) if (p + r) > 0 else 0.0  # Avoid division by zero
        for p, r in zip(precision, recall)
    ]

    return f1_scores


def _load_trained_model(wandb_project, model_type, default_path):
    """
    Helper function. Loads the best model checkpoint from Weights & Biases or local directory.
    First looks for specified project and pulls best model from project artifacts.
    If not found, loads the model from the default path.

    Args:
        wandb_project (str): The W&B project name.
        model_type (str): The type of model to load.
        default_path (str): Path to load local checkpoints if needed.

    Returns:
        tuple: The loaded model and processor.
    """
    logger.info("[INFERENCE] - [MODEL] - Loading best model from W&B...")
    
    try:
        run = wandb.init(project=wandb_project, entity="landeros-mgb")
        logger.debug(f"[INFERENCE] - [MODEL] - Successfully initialized W&B project: {wandb_project}")

        artifact_name =f'{wandb_project}/best_model:latest'
        artifact = run.use_artifact(artifact_name, type='model')
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, 'latest_checkpoint.pth')

        logger.info(f"[INFERENCE] - Successfully downloaded model from W&B project {wandb_project}.")
        logger.debug(f"[INFERENCE] - Model path: {model_path}")
    except Exception as e:
        logger.error(f"[INFERENCE] - [MODEL] - Failed to load model from W&B: {e}")
        logger.info(f"[INFERENCE] - [MODEL] - Loading local model from {default_path}...")
        model_path = default_path
    
    model, processor = load_model(model_type=model_type, num_labels=2, checkpoint_path=model_path, output_attentions=True)
    model.eval()
    return model, processor


def load_trained_model(wandb_project, default_model_type, default_path, device):
    """
    Loads the best model checkpoint from Weights & Biases or local directory.
    
    Args:
        wandb_project (str): The W&B project name.
        model_type (str): The type of model to load.
        default_path (str): Directory to load local checkpoints if needed.
    
    Returns:
        torch.nn.Module: The loaded model.
    """
    try:
        model, processor = _load_trained_model(wandb_project, default_model_type, default_path)
        model.eval()
        model.to(device)
    except Exception as e:
        logger.error(f"[INFERENCE] - [MODEL] - Failed to load model: {e}")
        sys.exit(1)
    return model


def joint_attention(attention_maps):
    """
    Computes joint attention by recursively propagating attention across layers.

    Args:
        attention_maps (torch.Tensor): Attention weights. [N_layer, batch_size, N_head, N_token, N_token].

    Returns:
        np.ndarray: Joint attention scores per input token (seq_len,).
    """
    logging.debug(f"[INFERENCE] - [Attention Map] - Performing joint attention computation...")

    attn_avg_heads, _ = attention_maps.max(dim=2)
    eye = torch.eye(attn_avg_heads.size(-1)).unsqueeze(0).unsqueeze(0).to(attn_avg_heads.device)
    attn_with_skip = attn_avg_heads + eye 
    
    logging.debug(f"[INFERENCE] - [Attention Map] - Attn with skip shape: {attn_with_skip.shape}")

    joint_attn = attn_with_skip[0]  # First layer
    for layer in range(1, attn_with_skip.shape[0]):
        joint_attn = torch.matmul(joint_attn, attn_with_skip[layer])

    joint_attn /= attn_with_skip.shape[0]
    joint_attn = joint_attn[0].mean(dim=0).detach().cpu().numpy()

    logging.debug(f"[INFERENCE] - [Attention Map] - Joint attention shape: {joint_attn.shape}")    
    return joint_attn


def compute_scores_and_attn(outputs, waveform):
    """
    Computes importance scores and joint attention from the model outputs.
    
    Args:
        outputs (dict): The model outputs.
        waveform (torch.Tensor): The audio waveform.

    Returns:
        np.ndarray: The importance scores. Of the same shape as the waveform.
        np.ndarray: joint attention scores.
    """
    logger.info(f"[INFERENCE] - [Attention Map] - Generating attention map...")
    try:
        outputs.logits.mean().backward()
        importance_scores = waveform.grad.abs().mean(dim=0).squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"[INFERENCE] - [Attention Map] - Failed to generate gradients: {e}")
    try:
        attentions = outputs.attentions
        if attentions is not None:
            logger.info("[INFERENCE] - [Attention Map] - Extracted attention weights.")
        else:
            logger.warning("[INFERENCE] - [Attention Map] - Attention weights not available.")
    except Exception as e:
        logger.error(f"[INFERENCE] - [Attention Map] - Failed to generate attention map: {e}")
    return importance_scores, joint_attention(torch.stack(attentions))


def generate_vis_map(waveform, importance_scores, attn_rollout, audio_path, preds):
    """
    Generates a visualization map of the audio waveform with importance scores and attention weights.
    Correlates the audio waveform with the ground truth annotations.
    Saves the visualization map to a PNG file based on the audio file name.

    Args:
        waveform (torch.Tensor): The audio waveform.
        importance_scores (np.ndarray): The importance scores. Will have length of waveform.
        attentions (list): The attention weights. Token-level attentions. [N_layer, batch_size, N_head, N_token, N_token].
        audio_path (str): Path to the audio file.
        preds (np.ndarray): The model predictions.
    """
    title = os.path.basename(audio_path)
    annotations = audio_path.replace(".wav", ".txt")
    annotations_df = pd.read_csv(annotations, sep="\t", names=ANNOTATIONS)
    crackles = annotations_df['crackles'].max()
    wheezes = annotations_df['wheezes'].max()
    gt = "+C" if (crackles and not wheezes) else "+W" if (wheezes and not crackles) else "+CW" if (crackles and wheezes) else "–"
    title += f" ({gt})"
    title += f" - preds: ({preds[0][0]:.2f}, {preds[0][1]:.2f})"
    print(crackles, wheezes)
    print(preds>0.5)


    save_path = audio_path.replace(".wav", "_vis.png")
    logger.info(f"[INFERENCE] - [Attention Map] - Saving visualization map to: {save_path}")

    waveform = np.squeeze(waveform.detach().numpy())

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    num_frames = len(waveform)
    time_axis = np.arange(0, num_frames).astype(float) / SAMPLE_RATE
    ax.plot(time_axis, waveform, linewidth=1, color=[.2, .2, .2], alpha=0.8, label="Auscultation Waveform")

    window_size = 10000
    importance_scores_vis = importance_scores
    importance_scores_vis = np.convolve(importance_scores_vis, np.ones(window_size)/window_size, mode='same')
    importance_scores_vis = np.clip(importance_scores_vis, 0, 1.0)
    importance_scores_vis = (importance_scores_vis - importance_scores_vis.min()) / (importance_scores_vis.max() - importance_scores_vis.min())
    ax.fill_between(time_axis, 0, 1, color='yellow', where=importance_scores_vis > 0.5, alpha=0.5, label="Importance Scores")    
    
    window_size = 1000
    scale_factor = float(len(waveform)) / len(attn_rollout)
    attn_rollout = zoom(attn_rollout, scale_factor, order=0)[:len(waveform)]
    attn_rollout = np.convolve(attn_rollout, np.ones(window_size)/window_size, mode='same')
    attn_rollout = (attn_rollout - attn_rollout.min()) / (attn_rollout.max() - attn_rollout.min())
    ax.fill_between(time_axis, 0, 1, color='orange', where=attn_rollout > 0.5, alpha=0.5, label="Attn Scores")    

    ax.legend(loc='upper right', fontsize=10)


    for idx, row in annotations_df.iterrows():
        start, end = row['start_time'], row['end_time']
        crackles, wheezes = bool(row['crackles']), bool(row['wheezes'])
        color = 'red' if (crackles and not wheezes) else 'blue' if (wheezes and not crackles) else 'purple' if (crackles and wheezes) else 'gray'
        class_code = '+C' if (crackles and not wheezes) else '+W' if (wheezes and not crackles) else '+CW' if (crackles and wheezes) else '–'
        ax.fill_between(time_axis, -1, -0.9, color=color, where=((time_axis > start) & (time_axis < end)), alpha=0.5, label=f"Resp. Cycle {1 + idx} ({class_code})")

    ax.text(time_axis[-1]/2, -1.065, "Respiratory Cycles (red: +C, blue: +W, purple: +CW)", fontsize=8, ha='center', va='center', color='black', backgroundcolor=[1.0, 1.0, 1.0, 0.5])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)

    plt.savefig(save_path)


def run_inference_file(model, device, audio_path, generate_vis=False):
    """
    Runs inference on a single audio file.
    Loads the audio file, preprocesses it, and generates predictions.
    if generate_vis is True, also generates and saves interpretability visualizations.

    Args:
        model (torch.nn.Module): The trained model.
        device (torch.device): The device to run inference on.
        audio_path (str): Path to the audio file.
        generate_vis (bool): Whether to generate interpretability visualizations.
    Returns:
        dict: The prediction results. Contains file path, crackles, wheezes, and response time.
    """
    
    if not os.path.isfile(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")
    if not audio_path.endswith(".wav"):
        raise ValueError(f"Invalid audio file format: {audio_path}. Expected .wav file.")
    start_time = time()

    logger.info(f"[INFERENCE] - Processing audio file: {audio_path}")
    try:
        # Load and Preprocess
        waveform, sr = torchaudio.load(audio_path)
        waveform = preprocess_audio(waveform, sr, SAMPLE_RATE).to(device)
        logger.debug(f"[INFERENCE] - Waveform Stats:")
        logger.debug(f"[INFERENCE] -  - Shape: {waveform.shape}")
        logger.debug(f"[INFERENCE] -  - Mean: {waveform.mean()}, Min: {waveform.min()}, Max: {waveform.max()}")
        
        if generate_vis:
            waveform.requires_grad = True
            outputs = model(waveform, output_attentions=True)
        else:
            with torch.no_grad():
                outputs = model(waveform)
        
        predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f"[INFERENCE] - Completed inference for: {audio_path} in {elapsed_time:.2f} seconds")

        importance_scores = attentions = None
        if generate_vis:
            importance_scores, attentions = compute_scores_and_attn(outputs, waveform)
            generate_vis_map(waveform, importance_scores, attentions, audio_path, predictions)

        return {
            "audio_file": audio_path,
            "crackles": float(predictions[0][0]),
            "wheezes": float(predictions[0][1]),
            "response_time": elapsed_time
        }
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None


def run_inference(model, device, audio_path_or_dir, generate_vis=False):
    """
    Runs inference on the provided audio file or directory.

    Args:
        model (torch.nn.Module): The trained model.
        device (torch.device): The device to run inference on
        audio_path_or_dir (str): Path to the audio file or directory.
        generate_vis (bool): Whether to generate interpretability visualizations.
    
    Returns:
        dict: The prediction results. Contains file path, crackles, wheezes, and response time.
    """
    audio_path_or_dir = os.path.normpath(audio_path_or_dir)

    if os.path.isfile(audio_path_or_dir):
        return {audio_path_or_dir: run_inference_file(model, device, audio_path_or_dir, generate_vis=generate_vis)}
    
    elif os.path.isdir(audio_path_or_dir):
        if not os.listdir(audio_path_or_dir) or not any(f.endswith(".wav") for f in os.listdir(audio_path_or_dir)):
            raise ValueError(f"Directory {audio_path_or_dir} is empty or contains no .wav files.")
        
        predictions = {}
        for f in os.listdir(audio_path_or_dir):
            if f.endswith(".wav"):
                file_path = os.path.join(audio_path_or_dir, f)
                file_preds = run_inference_file(model, device, file_path, generate_vis=generate_vis)
                if file_preds is not None:
                    predictions[file_path] = file_preds
        return predictions
    else:
        raise ValueError(f"Invalid audio path: {audio_path_or_dir}. Must be a file or directory.")
    

def calculate_metrics(metrics, predictions, labels):
    """
    Evaluates the inference results against ground truth labels.
    
    Args:
        metrics (torchmetrics.MetricCollection): The metrics object.
        predictions (torch.Tensor): The predicted values.
        labels (torch.Tensor): The ground truth values.
    
    Returns:
        dict: The computed metrics.
    """
    metrics.update(predictions > 0.5, labels)
    results = metrics.compute()

    logger.info(f"[INFERENCE] - [METRICS] - Results:")
    for key, value in results.items():
        logger.info(f"[INFERENCE] - [METRICS] -  - {key}: {value.item() if value.numel() == 1 else value.tolist()}")
    return results
    

def evaluate_inference(predictions, audio_path_or_dir):
    """
    Evaluates the inference results against ground truth labels.
    Ground truth labels are loaded from the same directory as the audio files.
    
    Args:
        predictions (dict): The prediction results.
        audio_path_or_dir (str): Path to the audio file or directory.
    
    Returns:
        dict: The computed metrics.
        torch.Tensor: The predicted values.
        torch.Tensor: The ground truth values.
    """
    base_dir = os.path.dirname(audio_path_or_dir) if os.path.isfile(audio_path_or_dir) else audio_path_or_dir
    label_path = os.path.join(base_dir, "labels.json")
    if not os.path.isfile(label_path):
        raise ValueError(f"Label file not found: {label_path}")
    logger.info(f"[INFERENCE] - [METRICS] - Loading labels from: {label_path}")

    preds_dict = flatten_preds(predictions)
    labels_dict = load_inference_labels(label_path)

    preds, labels = [], []
    for filename, values in labels_dict.items():
        filename = filename.replace("./", "")
        preds.append(preds_dict[filename])
        labels.append(values)
    preds = torch.tensor(preds, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    logger.info("[INFERENCE] - [METRICS] - Calculating metrics...")
    metrics = setup_metrics(num_labels=preds.shape[1])
    results = calculate_metrics(metrics, preds, labels)

    logger.info(f"[INFERENCE] - [METRICS] - Results:")
    for key, value in results.items():
        logger.info(f"[INFERENCE] - [METRICS] -  - {key}: {value.item() if value.numel() == 1 else value.tolist()}")
    
    return results, preds, labels


def load_inference_labels(label_path):
    """
    Loads the ground truth labels for inference evaluation.

    Args:
        label_path (str): Path to the label file.

    Returns:
        dict: The ground truth labels.
    """
    if not os.path.isfile(label_path):
        raise ValueError(f"Label file not found: {label_path}")
    
    with open(label_path, "r") as f:
        labels = json.load(f)

    converted_labels = {filename: [values['crackles'], values['wheezes']] for filename, values in labels.items()}
    return converted_labels


def flatten_preds(predictions):
    """
    Flattens the prediction results for evaluation.

    Args:
        predictions (dict): The prediction results.

    Returns:
        dict: The flattened prediction results.
    """
    return {filename: [values['crackles'], values['wheezes']] for filename, values in predictions.items()}


def save_predictions(predictions, audio_path_or_dir):
    """
    Saves predictions to a JSON file.
    
    Args:
        predictions (dict): The prediction results.
        project_name (str): The W&B project name.
        audio_path_or_dir (str): Path or directory to the audio file(s)
    """
    base_dir = os.path.dirname(audio_path_or_dir) if os.path.isfile(audio_path_or_dir) else audio_path_or_dir
    output_path = os.path.join(base_dir, f"predictions.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    logger.info(f"[INFERENCE] - Predictions saved to: {output_path}")


def save_metrics(metrics, audio_path_or_dir):
    """
    Saves metrics to a JSON file.
    
    Args:
        metrics (dict): The metric results.
        audio_path_or_dir (str): Path or directory to the audio file(s)
    """
    base_dir = os.path.dirname(audio_path_or_dir) if os.path.isfile(audio_path_or_dir) else audio_path_or_dir
    output_path = os.path.join(base_dir, "inference_metrics.json")
    
    metrics = {key: value.item() if value.numel() == 1 else value.tolist() for key, value in metrics.items()}
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"[INFERENCE] - Metrics saved to: {output_path}")


def calculate_per_class_confusion_matrices(predictions, labels):
    """
    Calculates confusion matrices for each class separately.

    Args:
        predictions (torch.Tensor): Predicted values of shape (N, 2).
        labels (torch.Tensor): Ground truth values of shape (N, 2).

    Returns:
        dict: Confusion matrices for each class.
    """
    preds_binary = (predictions > 0.5).int().cpu().numpy()
    labels_numpy = labels.int().cpu().numpy()

    class_names = ["crackles", "wheezes"]
    confusion_matrices = {}

    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(labels_numpy[:, i], preds_binary[:, i])
        confusion_matrices[class_name] = cm.tolist()  # Convert to list for easier logging

        logger.info(f"[INFERENCE] - [CONFUSION MATRIX] - {class_name}:")
        logger.info(f"\n{cm}")

    return confusion_matrices


def main():
    """
    Main function for running the inference application.
    Will execute these steps:
    - Parse command-line arguments.
    - Load the trained model.
    - Perform inference on the provided audio file or the files in the audio directory (audio_path).
    - Evaluate the inference metrics if specified.
    - Save the predictions and metrics to JSON files at the audio directory.
    """
    logger.info("="*50)
    logger.info("[INFERENCE] - Starting inference application...")
    logger.info("="*50)
    parser = argparse.ArgumentParser(description="Respiratory Sound Crackles and Wheeze Detection.")
    parser.add_argument("audio_path", type=str, help="Path to the test WAV file.")
    parser.add_argument("--model_path", type=str, default="./checkpoints/inference_model.pth", help="Path to default trained model.")
    parser.add_argument("--evaluate_metrics", action="store_true", help="Evaluate inference metrics.")
    parser.add_argument("--project_name", type=str, default=WANDB_PROJECT, help="W&B project name.")
    parser.add_argument("--generate_vis", action="store_true", help="Visualize attention maps.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")


    # Log parser arguments
    for arg, value in vars(args).items():
        logger.info(f"[INFERENCE] - Argument {arg}: {value}")    

    # Set device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[INFERENCE] - [MODEL] - Using device: {device}")
    model = load_trained_model(args.project_name, DEFAULT_MODEL_TYPE, args.model_path, device)

    # Perform inference
    try:
        preds = run_inference(model, device, args.audio_path, generate_vis=args.generate_vis)
        save_predictions(preds, args.audio_path)

        if args.evaluate_metrics:
            results, preds, labels = evaluate_inference(preds, args.audio_path)
            save_metrics(results, args.audio_path)
        else:
            logger.info("[INFERENCE] - Metrics evaluation disabled.")
        logger.info("[INFERENCE] - Inference completed.")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
