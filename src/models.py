"""
models.py
Author: landeros10
Date: 2025-01-25
Description: Contains model architectures and loading functions.

"""

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import os
import hashlib

from src.logger import logger

# Constants
BASE_MODEL = "facebook/wav2vec2-base"


def calculate_model_checksum(model):
    """
    Calculate a checksum (SHA256 hash) of the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to generate a checksum for.

    Returns:
        str: The SHA256 hash of the model parameters.
    """
    hasher = hashlib.sha256()
    for param_tensor in model.state_dict().values():
        hasher.update(param_tensor.cpu().numpy().tobytes())
    return hasher.hexdigest()


def load_wav2vec2_model(checkpoint_path=None, num_labels=2, output_attentions=False):
    """
    Load a Wav2Vec2 model for sequence classification.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to None.
        num_labels (int): Number of output labels. Defaults to 2.

    Returns:
        model (torch.nn.Module): Loaded Wav2Vec2 model.
        processor (Wav2Vec2Processor): Wav2Vec2 processor for pre-processing audio data.
    """

    logger.info("Loading Wav2Vec2 model")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        output_attentions=output_attentions,
        problem_type="multi_label_classification",
    )

    # Freeze feature extraction layers
    logger.debug("Freezing feature extraction layers")
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False

    initial_checksum = calculate_model_checksum(model)
    logger.debug(f"Initial model checksum: {initial_checksum}")
    
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            logger.info(f"Restoring model from checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)

            updated_checksum = calculate_model_checksum(model)
            logger.debug(f"Updated model checksum: {updated_checksum}")
        else:
            logger.debug(f"Checkpoint file not found: {checkpoint_path}")
    else:
        logger.debug("No checkpoint path provided")

    return model, processor


# def load_resnet_model(checkpoint_path=None, num_labels=2):
#     # TODO
#     model = ResNetModel(num_classes=num_labels)
#     if checkpoint_path:
#         model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
#     return model


def load_model(model_type='wav2vec', checkpoint_path=None, num_labels=2, output_attentions=False):
    """
    Load a model based on the specified type.

    Args:
        model_type (str): Type of model to load ('wav2vec' or 'resnet').
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to None.
        num_labels (int): Number of output labels. Defaults to 2.

    Returns:
        model (torch.nn.Module): Loaded model.
        processor (optional): Processor for the model (if applicable).
    """
    if model_type == 'wav2vec':
        return load_wav2vec2_model(checkpoint_path=checkpoint_path, num_labels=num_labels, output_attentions=output_attentions)
    else:
        raise ValueError(f"Unknown model type: {model_type}")