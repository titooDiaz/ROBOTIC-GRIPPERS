# ROBOTIC GRIPPERS

## Overview

## Features

## Requirements

The project depends on Python 3 and the packages listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Data collection

To collect new gesture samples, run the data collector and record a label from the webcam:

```bash
python -m src.data_collector --gesture name --samples 300
```

Available gesture labels are defined in `src.data_collector.py`:

- "wave",
- "ok",
- "two",
- "three"

Use the space bar to start or pause recording, and press `Q` to exit.

## Training

Once your dataset is ready, train the model:

```bash
python -m src.train
```

This script:

- loads the dataset from `data/raw/gestures.csv`
- scales the input features
- trains a small neural network
- saves the best model checkpoint to `models/gesture_model.pth`
- saves the scaler to `models/scaler.joblib`
- stores training metrics and visualization plots in `models/`

## Live inference

To run the webcam recognition demo:

```bash
python src.main.py
```

Press `Q` in the video window to quit.

## Example workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect training data
python -m src.data_collector --gesture wave --samples 300
python -m src.data_collector --gesture ok --samples 300
python -m src.data_collector --gesture two --samples 300
python -m src.data_collector --gesture three --samples 300

# 3. Train the model
python -m src.train

# 4. Run real-time recognition
python -m src.main.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
