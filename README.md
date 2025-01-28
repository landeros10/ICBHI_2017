# [ICBHI_2017] Identifying Wheezes and Crackles in Raw Respiratory Recordings

## Overview

This project aims to classify respiratory sounds using deep learning models, focusing on detecting wheezes and crackles from raw audio recordings. The system is built using PyTorch and Hugging Face Transformers and incorporates Weights & Biases (W&B) for tracking experiments.

## Project Structure

```
ICBHI_2017/
├── checkpoints/           # Model checkpoints
├── data/                  # Data directory
│   ├── test/               # Test dataset
├── logs/                  # Training logs
├── src/                   # Source code
│   ├── __init__.py
│   ├── data.py             # Data processing functions
│   ├── inference.py        # Inference pipeline
│   ├── logger.py           # Logging configuration
│   ├── models.py           # Model definitions and loading
│   ├── train.py            # Training pipeline
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt       # Required packages
```

## Prerequisites

### System Requirements

- Operating System: Linux, macOS, or Windows
- Python Version: 3.10
- Conda (Miniconda or Anaconda)

## Installation

```bash
  git clone https://github.com/landeros10/ICBHI_2017
  cd ICBHI_2017
```


### Step 1: Create a Conda Environment

```bash
conda create -n resp_sounds python=3.10 -y
conda activate resp_sounds
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Inference

To perform inference on test data, you must provide either the path to a single WAV file or a directory containing multiple WAV files. The script automatically determines the base directory `data_dir`. Additionally, the script looks for a `labels.json` file in the same directory to evaluate model predictions against ground truth labels if the `--evaluate_metrics` tag is used. Predictions and evaluation metrics are saved in `data_dir` as predictions.json or `inference_metrics.json`, respectively.

```bash
python src/inference.py audio_path [--model_path MODEL_PATH] [--evaluate_metrics] [--generate_vis]
```

### Example Usage:

#### Inference on a Single WAV File with Ground Truth Evaluation

```bash
python src/inference.py ./data_dir/sample.wav --model_path ./checkpoints/inference_model.pth --evaluate_metrics
```

#### Inference on a Directory of WAV Files

```bash
python src/inference.py ./data_dir/ --generate_vis
```


### Arguments:

- `audio_path`: The path to the input data. This can be a single WAV file (e.g., `./data_dir/sample.wav`) or a directory containing multiple WAV files (e.g., `./data_dir/`).
- `--evaluate_metrics`: Optional flag to evaluate the model's predictions against ground truth labels. Requires a `labels.json` file in the `data_dir`.
- `--model_path`: Path to the trained model used for inference. Defaults to `./checkpoints/inference_model.pth`.
-`--generate_vis`: Optional flag to visualize attention maps during inference. Visualizations are saved in the `data_dir` on a per-file basis.

## Results

After inference, per-file predictions are stored in `predictions.json`, inference metrics under `inference_metrics.json` and visualizations as `[audio_file_name].png`. All are saved in the same directory as the test data, `data_dir`.

### Structure of `labels.json` and `predictions.json`

Both `labels.json` and `predictions.json` share the same structure. Each key corresponds to the test audio file path, and the value is a dictionary indicating the presence or absence of crackles and wheezes as binary integer values.
#### Example Structure:

```json
{
    "./data/test/sample1.wav": {
        "crackles": 0,
        "wheezes": 0
    },
    "./data/test/sample2.wav": {
        "crackles": 1,
        "wheezes": 0
    },
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

