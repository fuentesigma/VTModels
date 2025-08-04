import os
import glob
from multiprocessing import Pool
# ------------------------------------------------------/
import h5py
import numpy as np
import pandas as pd
# ------------------------------------------------------/
# Change directory accordingly
input_folder = "/Volumes/LaCie/HeartProject/VT-holter"

# Leads - all data respects this format 
ecg_leads = ["i (mV)", "ii (mV)", "iii (mV)", "avr (mV)", "avl (mV)", "avf (mV)", "v1 (mV)", "v2 (mV)", "v3 (mV)", "v4 (mV)", "v5 (mV)", "v6 (mV)"]

"""
    N: Normal beat (normal sinus rhythm)
    F: Fusion or noisy beat (a mix of normal and abnormal beats)
    V: Ventricular ectopic beat (premature ventricular contraction)
    S: Supraventricular beat (from atria or AV node)
    A: Atrial beat
    Q: Unknown or questionable beat
"""

class Bucket:
    def __init__(self, input_folder, fs=128, leads=None):
        self.fs = fs
        self.input_folder = input_folder
        self.leads = leads if leads is not None else ecg_leads

    def ecg(self, file_number):
        search_pattern = os.path.join(self.input_folder, f"{file_number}-*.txt")
        matching_files = [f for f in glob.glob(search_pattern) if "_RPointProperty" not in os.path.basename(f)]

        if not matching_files:
            print(f"No ECG file found for {file_number}.")
            return None
        correct_columns = ["Time (sec)"] + ecg_leads

        try:
            data = pd.read_csv(matching_files[0], sep="\s+", skiprows=3, names=correct_columns, usecols=correct_columns)
            voltage = data[self.leads].values
            data[self.leads] = self.broom(voltage)
            return data
        
        except Exception as e:
            print(f"Error processing {matching_files[0]}: {e}")
            return None

    def annotations(self, file_number):
        search_pattern = os.path.join(self.input_folder, f"{file_number}-*_RPointProperty.txt")
        matching_files = glob.glob(search_pattern)

        if not matching_files:
            print(f"No annotation file found for {file_number}.")
            return None
        
        annotations = []
        try:
            with open(matching_files[0], "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        time_str, label = line.strip().split(":")
                        try:
                            annotations.append({"Time": float(time_str), "Label": label.strip()})
                        except ValueError:
                            print(f"Invalid annotation: {line.strip()}")
            return pd.DataFrame(annotations) if annotations else pd.DataFrame(columns=["Time", "Label"])
        
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return None

    def broom(self, data):
        # Import cleaner class
        from ecgCleaner import ECGProcessor

        # Delegate ECG cleaning to ECGProcessor
        processor = ECGProcessor(lead_names=self.leads, fs=self.fs)

        # Run the pipeline
        # signal, *_ = processor.pipeline(data.copy(), action="mask", plot_summary=False)

        # Channel assessment
        # signal, *_ = processor.wizard(signal)

        signal = processor.wavelets(data.copy())
        
        return signal

    def fillme(self, file_number):
        ecg = self.ecg(file_number)
        ann = self.annotations(file_number)

        if ecg is None or ann is None:
            return ecg
        
        # Align using pandas.merge_asof
        ecg_sorted = ecg.sort_values("Time (sec)")
        ann_sorted = ann.sort_values("Time")
        
        merged = pd.merge_asof(
            ecg_sorted,
            ann_sorted,
            left_on="Time (sec)",
            right_on="Time",
            direction="nearest",
            tolerance=1 / self.fs
        )
        
        ecg["Annotation"] = merged["Label"].values

        return ecg

class Glypho:
    def __init__(self, output_file, fs=128, ds=2, margin=600, leads=None, start_offset=300):
        self.fs = fs
        self.ds = ds
        self.margin = margin
        self.leads = leads if leads is not None else ecg_leads
        
        # Default recording start offset in seconds (5 minutes)
        self.start_offset = start_offset

        if not os.path.exists("VT-data"):
            os.makedirs("VT-data")
        self.output_file = os.path.join("VT-data", f"Patient_{output_file}.h5")

    def labelling(self, data, bpm=100):
        # Extract all annotation times and labels in order
        annotations = data[["Time (sec)", "Annotation"]].dropna().reset_index(drop=True)
        times = annotations["Time (sec)"].values
        labels = annotations["Annotation"].values

        episodes = []
        start_idx = None

        # Scan for runs of consecutive 'V' labels
        for idx, label in enumerate(labels):
            if label == "V":
                if start_idx is None:
                    start_idx = idx
            else:
                # End of a V-run
                if start_idx is not None:
                    run_len = idx - start_idx
                    if run_len >= 3:
                        run_times = times[start_idx:idx]
                        intervals = np.diff(run_times)
                        # Check if all intervals satisfy the bpm threshold
                        if np.all(intervals <= 60.0 / bpm):
                            episodes.append((run_times[0], run_times[-1]))
                    start_idx = None

        # Check for a V-run that continues to the last annotation
        if start_idx is not None:
            run_len = len(labels) - start_idx
            if run_len >= 3:
                run_times = times[start_idx:]
                intervals = np.diff(run_times)
                if np.all(intervals <= 60.0 / bpm):
                    episodes.append((run_times[0], run_times[-1]))

        if not episodes:
            return None

        # Use first VT episode
        start_time, end_time = episodes[0]

        # Define trimming window: 2h before onset, 20min after end, not before start_offset
        segment_start = max(self.start_offset, start_time - 2 * 3600.0)
        segment_end = end_time + 20 * 60.0

        return (segment_start, start_time, end_time, segment_end)

    def inscribe(self, data, bpm=100):
        # Extract full time vector and 12-lead ECG matrix
        times = data["Time (sec)"].values
        ecg_matrix = data[self.leads].values

        # Determine VT trimming window
        segment = self.labelling(data, bpm)

        if segment is None:
            print(f"No VT episode detected for {self.output_file}.")
            return
        
        segment_start, event_start, event_end, segment_end = segment

        # Crop to the trimming window
        mask = (times >= segment_start) & (times <= segment_end)
        times_crop = times[mask]
        ecg_crop = ecg_matrix[mask, :]

        # Downsample ECG and time vector from fs to fs / ds
        if self.ds > 1:
            times_ds = times_crop[::self.ds]
            ecg_ds = ecg_crop[::self.ds, :]
        else:
            times_ds = times_crop
            ecg_ds = ecg_crop

        # Normalise ECG leads to zero mean and unit variance
        means = np.mean(ecg_ds, axis=0)
        stds = np.std(ecg_ds, axis=0)
        ecg_ds = (ecg_ds - means) / stds

        # Create VT binary labels aligned with downsampled times
        vt_label = np.zeros(times_ds.shape, dtype=int)
        if event_start is not None and event_end is not None:
            vt_mask = (times_ds >= event_start) & (times_ds <= event_end)
            vt_label[vt_mask] = 1

        # Combine into a single data matrix: time, 12 leads, VT label
        data_matrix = np.hstack([times_ds.reshape(-1, 1), ecg_ds, vt_label.reshape(-1, 1)])

        # Write trimmed ECG and VT labels to HDF5
        with h5py.File(self.output_file, "w") as f:
            f.create_dataset("data", data=data_matrix)

        print(f"Exported ECG + labels to {self.output_file}")

def triage(patient):
    # Load and align ECG with annotations
    loader = Bucket(input_folder=input_folder)
    data = loader.fillme(patient)
    
    if data is None:
        return

    # Export cleaned ECG and discrete annotations
    perseus = Glypho(output_file=patient, ds=4)
    perseus.inscribe(data)

def visdebugger(patient=1, a=5):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fpath = os.path.join("VT-data", f"Patient_{patient}.h5")

    # Load processed data
    with h5py.File(fpath, "r") as f:
        matrix = f["data"][:]
    times = matrix[:, 0]
    ecg = matrix[:, 1:13]
    vt = matrix[:, 13]

    # Identify first VT onset
    vt_indices = np.where(vt > 0)[0]
    if vt_indices.size == 0:
        print(f"No VT episode detected for patient {patient}.")
        return
    onset_idx = vt_indices[0]
    onset_time = times[onset_idx]

    # Define 5 seconds before and after window
    window_mask = (times >= onset_time - a) & (times <= onset_time + a)
    times_win = times[window_mask]
    ecg_win = ecg[window_mask]
    vt_win = vt[window_mask]

    # Plot ECG leads in blue with vertical offsets
    offsets = np.arange(ecg.shape[1]) * 5.0

    fig, ax = plt.subplots()
    for i in range(ecg_win.shape[1]):
        ax.plot(times_win, ecg_win[:, i] + offsets[i])

    # Highlight VT event window in red
    vt_window_indices = np.where(vt_win > 0)[0]
    if vt_window_indices.size > 0:
        vt_start = times_win[vt_window_indices[0]]
        vt_end = times_win[vt_window_indices[-1]]
        ax.axvspan(vt_start, vt_end, color='red', alpha=0.3)

    # Format x-axis as HHMMSS with at most 5 ticks
    def hhmmss(x, pos):
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(hhmmss))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

    # Label y-axis with ECG lead names
    ax.set_yticks(offsets)
    ax.set_yticklabels(ecg_leads)

    ax.set_xlabel("Time")
    plt.show()

def main():
    # Gather all files with pattern: "<file_number>-*.txt"
    pattern = os.path.join(input_folder, "*-*.txt")
    nfiles = [os.path.basename(f) for f in glob.glob(pattern) if "_RPointProperty" not in f]

    # Extract the distinct file numbers to define the cohort
    cohort = sorted({fname.split("-", 1)[0] for fname in nfiles})

    # Load multiprocessing
    with Pool(processes=4) as pool:
        pool.map(triage, cohort)

if __name__ == "__main__":
    # ////////////////////
    main()
    #      _______
    #     /       \
    #    |  O   O  |
    #    |    ^    |
    #     \  ~~~  /
    #      \_____/
    #       _|_|_
    #      |     |
    #      |     | j. fuentes