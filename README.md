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
- GPU: Optional (Recommended for faster training)
- Conda (Miniconda or Anaconda)
- Disk Space: At least 10GB for data and model checkpoints

## Installation

```bash
  git clone https://github.com/your-repo/respiratory-classification.git
  cd respiratory-classification
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

To perform inference on test data:

```bash
python src/inference.py ./data/test/ \
    --evaluate_metrics
```

### Arguments:

- `audio_path`: Path to the directory containing test audio files.
- `--evaluate_metrics`: Evaluate the model's predictions against ground truth.

## Results

After inference, results and visualizations are saved in the same directory as the test data. Metrics such as accuracy, precision, recall, and F1 scores are logged.

## Contributing

If you'd like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

