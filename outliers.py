import os
import glob
import h5py
import numpy as np
from sklearn.ensemble import IsolationForest

def extract_features(file_path):
    """
    Opens an HDF5 file of shape (T, 14): [time, 12 leads, VT label],
    and returns a feature vector consisting of lead-wise means, standard deviations,
    and selected percentiles.
    """
    with h5py.File(file_path, "r") as f:
        data = f["data"][:]
    ecg = data[:, 1:-1]

    # Compute mean and std for each lead
    means = np.mean(ecg, axis=0)
    stds  = np.std(ecg, axis=0)

    # Compute 5th, 25th, 50th, 75th, 95th percentiles
    percentiles = np.percentile(ecg, [5, 25, 50, 75, 95], axis=0).T.flatten()
    return np.concatenate([means, stds, percentiles])

def detect_and_rename(directory="VT-data", contamination=0.05, random_state=0):
    """
    Scans all Patient_*.h5 in the given directory, extracts features,
    fits an Isolation Forest, and renames any outlying file by appending '_o'.
    """
    pattern = os.path.join(directory, "Patient_*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found in {directory}.")
        return

    # Build feature matrix
    feature_list = [extract_features(fp) for fp in files]
    X = np.vstack(feature_list)

    # Fit the Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=random_state)

    # I use -1 for outliers, +1 for inliers
    labels = clf.fit_predict(X)

    # Rename outliers
    for path, lbl in zip(files, labels):
        if lbl == -1:
            base, ext = os.path.splitext(path)
            new_name = f"{base}_o{ext}"
            os.rename(path, new_name)
            print(f"Marked outlier: '{os.path.basename(path)}' → '{os.path.basename(new_name)}'")

if __name__ == "__main__":
    detect_and_rename()



    (vtlab) W:\VT project> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple h5py numpy wandb pywt matplotlib scikit-learn scipy pandas patool