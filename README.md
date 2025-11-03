# Ventricular Tachycardia Early Warning System

This repository implements a pipeline for training a deep learning model, **VTATTEND**, designed for early prediction of ventricular tachycardia (VT) using multi-lead ECG time series data. The project includes data preprocessing, outlier handling, self-supervised pretraining, model training, and final evaluation.

# Running `run_experiments.py` (NEW)

This document explains how to execute `run_experiments.py` and what other parts of the codebase are required. It is intended for internal use in this private repository.

## Purpose of `run_experiments.py`

The script `run_experiments.py` is a thin launcher around the training engine implemented in `beatbotLite.py`. It builds a cohort of patients from the HDF5 data files, defines a training and evaluation split, sets a few key hyperparameters, and then calls `run_training` from `beatbotLite`.

In other words, you do not normally call `beatbotLite.py` directly for day to day experiments. Instead, you run `run_experiments.py`, which prepares the arguments and invokes the training pipeline for you.

## Code pieces you need

In order to run `run_experiments.py` successfully, you need at least the following pieces in place.

First, you need `run_experiments.py` itself, located in the repository root (or in whatever directory the project is organised around). The script imports two names from `beatbotLite`:

```python
from beatbotLite import run_training, build_parser

---

## File Summary (outdated)

### `dataCleaner.py`: ECG Preprocessing and Segmentation

This script is the **first step** in the VTATTEND pipeline. It processes raw ECG signal data and associated annotations into a clean, downsampled, and trimmed dataset suitable for training. The final output is written in HDF5 format.

---

- ECG waveform files in plain text, expected to follow the naming format:  
  `<PatientID>-<RecordingID>.txt` (e.g., `12.txt`)
  
- Annotation files for each recording, named as:  
  `<PatientID>-<RecordingID>_RPointProperty.txt`

The script assumes all raw data files are located in the folder defined by the variable `input_folder`.  
You must modify this variable to point to the correct local path where your dataset resides.

```python
input_folder = "/path/to/raw/ECG-data"
```

---

1. **Loads ECG waveform data** using a 12-lead configuration.
2. **Cleans each ECG signal** via the `ECGProcessor` class from `ecgCleaner.py`:
   - Currently, wavelet denoising is applied.
3. **Loads beat annotations** (labels such as N, V, F, etc.).
4. **Aligns annotations with ECG time series** using `pandas.merge_asof`.
5. **Identifies ventricular tachycardia (VT) episodes** using simple clinical rules:
   - At least 3 consecutive ‘V’ beats within a heart rate > 100 bpm.
6. **Trims the recording** to a window: 2 hours before VT onset and 20 minutes after VT termination.
7. **Downsamples the ECG** to reduce storage and compute cost.
8. **Normalises the ECG** (zero mean, unit variance per lead).
9. **Stores the result in an HDF5 file** containing:
   - Time vector
   - Normalised ECG matrix (12 leads)
   - Binary VT label vector (0 = no VT, 1 = VT)

---

The cleaned and labelled data are saved under a standardised folder:

```bash
VT-data/Patient_<PatientID>.h5
```

Each `.h5` file contains a dataset named `"data"` with shape `(T, 14)`, where:
- First column: time (in seconds)
- Columns 1–12: ECG channels (leads I, II, III, aVR, aVL, aVF, V1–V6)
- Last column: VT label (0 or 1)

This folder is automatically created if it does not exist. All downstream scripts (`beatbot.py`, `warning.py`) expect data to be in this location.

---

From the command line:

```bash
python dataCleaner.py
```

This script will:
- Automatically detect all patients based on the file naming convention
- Process the entire cohort in parallel using 4 processes
- Print messages for each processed file or if no VT episode was found

---

An optional visualisation function `visdebugger(patient_id)` is included, allowing inspection of ECG snippets around VT onset.  
This helps verify whether episodes have been correctly detected and labelled.

The script is deterministic and may be safely rerun after modifying:
- The ECG cleaning pipeline (e.g., filtering or masking)
- The VT detection criteria
- The `input_folder` path

To regenerate everything from scratch, simply delete the `VT-data` folder and rerun the script.

---

### ⚠️

- The definition of a VT episode is conservative and based on standard arrhythmia rules. Adjust the logic in `labelling()` if needed.
- If no VT episode is found for a patient, the file will be skipped.
- ECGs are assumed to be sampled at 128 Hz; change `fs` in `Bucket` or `Glypho` classes if different.

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
