# Ventricular Tachycardia Early Warning System

This repository implements a pipeline for training a deep learning model, **VTATTEND**, designed for early prediction of ventricular tachycardia (VT) using multi-lead ECG time series data. The project includes data preprocessing, outlier handling, self-supervised pretraining, model training, and final evaluation.

---

## Pipeline Overview

The training workflow consists of the following steps:

1. **Preprocess the raw ECG data**
2. **(Optionally) remove patient outliers**
3. **Train the VTATTEND model**
4. **(Optionally) run the post-training early warning system**

---

## File Summary

### `dataCleaner.py`
This script processes the raw ECG data and converts it into HDF5 files compatible with the training model. It handles segmenting, resampling, and formatting.

**Run this first.** It must be executed before any model training can take place.

---

### `ecgCleaner.py`
Called internally by `dataCleaner.py`. It contains lower-level signal cleaning routines such as baseline correction, filtering, and noise removal.

Users don't need to invoke this manually.

---

### `outliers.py`
Removes outlier patients from the preprocessed dataset (e.g., low-quality signals, inconsistent labels).  
**Optional but recommended.** Run it after `dataCleaner.py` and before training.

---

### `beatbot.py`
The main training script. It includes:

- Dataset loading and sliding window sampling (`BeatHarvest`)
- Model definition: a temporal convolutional network with self-attention (`VTATTEND`)
- Self-supervised pretraining and supervised fine-tuning (`Cardiobot`)
- Cross-validation and hyperparameter search (`Pipeline.controlpanel`)
- Final training and model export (`Pipeline.fulltrain`)
- A debug mode for rapid testing (`Pipeline.debug`)

---

### `warning.py`

**Run this after training is complete**, on the evaluation patients, to simulate deployment.

This script simulates real-time prediction on unseen ECG data using the best trained model checkpoint. It evaluates whether the model is capable of **raising early alarms** before the clinical onset of ventricular tachycardia (VT). It also provides interpretability via performance metrics, ROC curves, confusion matrices, and probability timelines.

---

Once the model has been trained (via `beatbot.py`), this script:

- Loads the best configuration (from `best_cfg.json` or Weights & Biases)
- Instantiates the model with the best hyperparameters
- Loads the trained model weights (`best_VTATTEND.pth`)
- Retrieves the test cohort for inference
- Generates smoothed probability outputs for each ECG window
- Detects if and when the predicted probability crosses a decision threshold for a sustained period (**dwell time**)
- Computes lead time, ROC AUC, confusion matrix, and standard classification metrics
- Produces visualisations for diagnostic and validation purposes

---

The logic is encapsulated in two classes:

- **`RingRing`**: Core inference and alarm detection logic. It performs sliding window inference, prediction smoothing, and finds sustained probability crossings.
- **`Monitor`**: Inherits from `RingRing` and adds diagnostic tools like accuracy, precision, F1-score, lead time computation, equilibrium threshold plotting (`eqpoint`), and a visual summary (`report`).

The script ends with a `main()` function that:

1. Loads the test set
2. Loads the model and weights
3. Runs the `Monitor.report()` method with a fixed threshold (default `0.37`)
4. Displays and saves diagnostic plots

---

Once `beatbot.py` has finished training:

```bash
python warning.py
```

This will evaluate the model on the test cohort and generate:

- A time series plot of predicted probabilities and true labels
- Onset and alarm times in `HH:MM:SS`
- ROC curve with AUC
- Confusion matrix heatmap
- A saved report as `eqpoint_point.pdf` (and optionally `report.pdf`)

If no alarms are raised or onset is undefined, it will report `None` for those entries.

- Default threshold is `0.37` and dwell time is 10 seconds.
- The smoothing window is Gaussian with σ = 30 seconds (converted to window units).
- Input data is expected in `.h5` format, preprocessed by `dataCleaner.py`.
- The script uses Apple MPS or CUDA if available, otherwise falls back to CPU.


---

## How to Run the Full Pipeline

1. **Prepare the Data**
   ```bash
   python dataCleaner.py
   ```

2. **(Optional) Remove Outliers**
   ```bash
   python outliers.py
   ```

3. **Train the Model**
   ```bash
   python beatbot.py
   ```

   This launches a Weights & Biases (W&B) hyperparameter sweep across cross-validation folds.  
   The best configuration will then be retrained with SSL pretraining and final evaluation.

4. **Run the Early Warning System**
   ```bash
   python warning.py
   ```

---

## Model: VTATTEND

**Architecture Summary**:
- A **Temporal Convolutional Network (TCN)** encodes the ECG signal.
- A **Multi-Head Self-Attention** layer models temporal dependencies.
- The model is trained in two stages:
  1. **Self-supervised pretraining** using a masked sequence reconstruction task.
  2. **Supervised fine-tuning** using binary classification of VT events.

---

## Output

- Trained model checkpoint saved as `best_VTATTEND.pth`
- Training logs and metrics visualised in W&B
- Precision, recall, and F1-score on validation and test cohorts

---

## Requirements

This code requires:
- `torch`
- `numpy`
- `h5py`
- `wandb`
- `scikit-learn`

Ensure you have a GPU or Apple silicon chip for efficient training.

---

## Citation

Work in progress. Please do not use this model for clinical decisions.
