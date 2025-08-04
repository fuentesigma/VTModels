import os
import json
import h5py
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# //////////////////////////////////////////////////////////////////////////
import torch
from wandb import Api
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, precision_score
# //////////////////////////////////////////////////////////////////////////
from beatbot import VTATTEND, BeatHarvest, SLIDING_WINDOW, evaluationCohort, validationCohort, input_dim
# ---------------------------------------------------------------------------------/
cfg_path = "best_cfg.json"

if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        best_cfg = json.load(f)

else:
    # Fetch best run from WandB and save config
    api = Api()
    entity = "jfuentesaguilar"
    project = "VTATTEND_"
    runs = api.runs(f"{entity}/{project}")
    best_run = min(runs, key=lambda r: r.summary.get("val_loss", float("inf")))
    best_cfg = best_run.config

    with open(cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)

# ---------------------------------------------------------------------------------/
class RingRing:
    def __init__(self, model, test_loader, smoothing_window=5):
        # Flag for smoothing window size
        self.smoothing_window = smoothing_window

        # Perform device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        # Retrieve full ECG data
        fpath = test_loader.dataset.files[0]

        with h5py.File(fpath, 'r', libver='latest', swmr=True) as f:
            matrix = f['data'][:]

        ecg = matrix[:, 1:13]
        self.labels = matrix[:, 13]

        # Time in seconds
        self.time = matrix[:, 0]

        # Get sliding window parameters
        self.sliding_window = test_loader.dataset.sliding_window
        self.step_size = test_loader.dataset.step_size
        self.horizon = test_loader.dataset.horizon

        # Store params for dwell-time alarm detection
        self.sampling_rate = 32
        
        # Number of samples in the ECG
        n_samples = ecg.shape[0]

        # Collect raw windowed predictions
        starts = []
        preds_list = []
        for start in range(0, n_samples - self.sliding_window - self.horizon + 1, self.step_size):
            window_data = ecg[start : start + self.sliding_window]
            window_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = torch.sigmoid(self.model(window_tensor)).squeeze(0).cpu().numpy()

            preds_list.append(preds)
            starts.append(start)

        # Collapse each window's sequence into a single score via mean
        self.pred_prob = np.array([preds.mean() for preds in preds_list])

        # Compute window end times in seconds
        self.time = (np.array(starts) + self.sliding_window) / self.sampling_rate

        # Ground-truth label per window
        self.labels = np.array(test_loader.dataset.sample_labels)

        if True:
            # Define smoothing parameters
            sigma_sec = 30

            # Convert smoothing sigma from seconds to window units
            window_interval = self.step_size / self.sampling_rate
            sigma_w = max(1, int(sigma_sec / window_interval))

            # Create Gaussian kernel
            kernel_size = 6 * sigma_w + 1
            kernel = np.exp(-0.5 * ((np.arange(kernel_size) - kernel_size//2) / sigma_w)**2)
            kernel /= kernel.sum()

            # Convolve predictions
            self.pred_prob = np.convolve(self.pred_prob, kernel, mode='same')

    def onset_time(self):
        event_indices = np.where(self.labels == 1)[0]
        return self.time[event_indices[0]] if event_indices.size > 0 else None
    
    def alarm_time(self, threshold, dwell_time=10):
        # Number of windows in which to sustain the threshold
        # [!] I put dwell_time in seconds
        dwell_samples = dwell_time * self.sampling_rate
        dwell_windows = max(1, int(np.ceil(dwell_samples / self.step_size)))

        # Slide over the window scores to find sustained crossing
        for i in range(len(self.pred_prob) - dwell_windows + 1):
            if np.all(self.pred_prob[i : i + dwell_windows] >= threshold):
                return float(self.time[i])
        return None

class Monitor(RingRing):
    def __init__(self, model, test_loader, smoothing_window=5):
        super().__init__(model, test_loader, smoothing_window)

    def scores(self, threshold=0.1):
        # Compute confusion matrix and metrics at a given threshold
        y_pred = (self.pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.labels, y_pred).ravel()

        # Calculate lead time
        alarm = self.alarm_time(threshold)
        onset = self.onset_time()

        if onset is None or alarm is None:
            lead_time = None
        else:
            lead_time = onset - alarm

        return {
            "Threshold": threshold,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Accuracy": accuracy_score(self.labels, y_pred),
            "Precision": precision_score(self.labels, y_pred, zero_division=0),
            "F1-score": f1_score(self.labels, y_pred, zero_division=0),
            "Lead Time": lead_time
        }

    def eqpoint(self, thresholds=np.linspace(0.1, 0.9, 100)):
        sens = []
        spec = []
        accr = []
        
        for thr in thresholds:
            metr = self.scores(thr)
            sens.append(metr['Sensitivity'])
            spec.append(metr['Specificity'])
            accr.append(metr['Accuracy'])

        sens = np.array(sens)
        spec = np.array(spec)
        accr = np.array(accr)

        # Find approximate intersection between sensitivity and specificity curves
        diff = np.abs(sens - spec)
        idx = np.argmin(diff)
        inter_thr = thresholds[idx]
        
        # Use the mean value since they are nearly equal at intersection
        inter_value = (sens[idx] + spec[idx]) / 2
        
        _, ax = plt.subplots(figsize=(8, 8))
        ax.plot(thresholds, sens, lw=3, color="#4287f5", label='Sensitivity',)
        ax.plot(thresholds, spec, lw=3, color="#b167db", label='Specificity')
        ax.plot(thresholds, accr, lw=3, color="#33de69", linestyle='--', label='Accuracy')

        # Intersection point and its coordinates
        ax.scatter(inter_thr, inter_value, color='#de3380', zorder=5, s=100, label=f'Equilibrium point ({inter_thr:.2f}, {inter_value:.2f})')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric Value')
        plt.legend()
        self.dressme(ax)
        plt.tight_layout()

        plt.savefig("eqpoint_point.pdf", format="pdf")

    def report(self, threshold=0.1):
        # Compute early warning values from the EWS object
        alarm = self.alarm_time(threshold)
        onset = self.onset_time()

        # Prepare ROC data using flattened test sets
        y_true_flat = self.labels.flatten()
        y_scores_flat = self.pred_prob.flatten()

        # Filter out non-finite prediction values
        mask = np.isfinite(y_scores_flat)
        y_true_flat = y_true_flat[mask]
        y_scores_flat = y_scores_flat[mask]
        fpr, tpr, _ = roc_curve(y_true_flat, y_scores_flat)
        roc_auc = auc(fpr, tpr)
        
        # Compute confusion matrix using the chosen threshold
        y_pred_flat = (y_scores_flat >= threshold).astype(int)
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        # Create a grid with 2 rows and 2 columns
        fig = plt.figure(tight_layout=True, figsize=(8, 8))
        gs = fig.add_gridspec(2, 2)
        
        # Convert onset and alarm from seconds to minutes and HH:MM:SS format
        def sec_to_hms(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        onset_hms = sec_to_hms(onset) if onset is not None else None
        alarm_hms = sec_to_hms(alarm) if alarm is not None else None

        # Early warning plot
        ax_ew = fig.add_subplot(gs[0, :])
        ax_ew.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: time.strftime('%H:%M:%S', time.gmtime(x))))
        ax_ew.step(self.time, self.labels, '-o', color="#e83875", alpha=0.8, lw=2, where="pre", label='True Label')
        ax_ew.plot(self.time, self.pred_prob, marker="o", color='#66d2e8', lw=2, label='Predicted Probability')
        ax_ew.axhline(threshold, color='gray', linestyle='--', lw=2, label=f'Threshold ({threshold:.2f})')
        ax_ew.axvline(onset, color='#e83875', linestyle='--', lw=4, label=f'Onset at t={onset_hms}')

        if alarm is not None:
            ax_ew.axvline(alarm, color='#6a49e3', linestyle='--', lw=4, label=f'Alarm at t={alarm_hms}')
            ax_ew.scatter(alarm, threshold, color='#6a49e3', s=100, zorder=5)
            
        ax_ew.set_xlabel("Time")
        ax_ew.set_ylabel("Probability / True Label")
        ax_ew.legend()
        self.dressme(ax_ew)
        
        # ROC curve plot
        ax_roc = fig.add_subplot(gs[1, 0])
        ax_roc.plot(fpr, tpr, lw=4, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], lw=2, linestyle='--')
        ax_roc.set_xlim([-0.05, 1.00])
        ax_roc.set_ylim([ 0.00, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc="lower right")
        self.dressme(ax_roc)
        
        # Confusion matrix plot
        ax_cm = fig.add_subplot(gs[1, 1])
        cmap = sns.diverging_palette(256, 0, as_cmap=True)
        sns.heatmap(cm, ax=ax_cm, annot=True, fmt='d', cmap=cmap, cbar=False, annot_kws={"fontsize":14, "color":"#444444"})
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        self.dressme(ax_cm)
        
        # # Table of metrics
        # # Build a table of metrics for a range of thresholds
        # import pandas as pd

        # thresholds = np.linspace(0.1, 0.9, 16)
        # allmetrics = [self.scores(thr) for thr in thresholds]
        # df_metrics = pd.DataFrame(allmetrics)

        # ax_table = fig.add_subplot(gs[2, :])
        # ax_table.axis('tight')
        # ax_table.axis('off')
        # table = ax_table.table(cellText=df_metrics.round(4).values, colLabels=df_metrics.columns, loc='center')
        # table.auto_set_font_size(False)
        # table.set_fontsize(12)
        # table.scale(1, 1.5)

        # for (row, col), cell in table.get_celld().items():
        #     cell.visible_edges = "LRB" if row == 0 else "LR"
                
        # plt.savefig("report.pdf")
        plt.show()

        # # Add a plot here of accuracy vs lead time
        # df_plot = df_metrics.dropna(subset=["Lead Time"])

        # if not df_plot.empty:
        #     fig2, ax2 = plt.subplots(figsize=(8, 6))
        #     ax2.plot(df_plot["Lead Time"], df_plot["Accuracy"], marker="o", lw=4)
        #     # Use log scale on both axes
        #     #ax2.set_xscale('log')
        #     #ax2.set_yscale('log')

        #     # Annotate points with threshold values, alternating offsets to reduce overlap
        #     for idx, (lt, acc, thr) in enumerate(zip(df_plot["Lead Time"], df_plot["Accuracy"], df_plot["Threshold"])):
        #         dx, dy = (5, 5) if idx % 2 == 0 else (5, -5)
        #         ax2.annotate(f"{thr:.2f}", xy=(lt, acc), xytext=(dx, dy), textcoords="offset points", fontsize=12)

        #     ax2.set_xlabel("Lead Time (sec)")
        #     ax2.set_ylabel("Accuracy")
        #     self.dressme(ax2)
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()
    
    @staticmethod
    def dressme(ax):
        fs = 18
        color = "#444444"
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)

        ax.tick_params(axis='both', colors=color, labelsize=fs)

        ax.xaxis.label.set_size(fs)
        ax.xaxis.label.set_color(color)

        ax.yaxis.label.set_size(fs)
        ax.yaxis.label.set_color(color)

        legend = ax.get_legend()
        if legend:
            plt.setp(legend.get_texts(), fontsize=fs, color=color)

# ---------------------------------------------------------------------------------/
def main():

    # Data loaders for evaluation
    evaluation_loader = BeatHarvest(
        [validationCohort[1]], 
        SLIDING_WINDOW, 
        best_cfg["horizon"], 
        best_cfg["batch_size"]
        ).push()
    
    # Instantiate model with best hyperparameters
    model = VTATTEND(
        input_dim=input_dim,
        channel_sizes=best_cfg["channel_sizes"],
        kernel_size=best_cfg["kernel_size"],
        num_heads=best_cfg["num_heads"],
        dropout=best_cfg["dropout"],
        horizon=best_cfg["horizon"]
    )

    # Load trained model weights
    model.load_state_dict(torch.load("best_VTATTEND.pth", map_location="cpu"))

    # Execute early warning evaluation
    monitor = Monitor(model, evaluation_loader)
    monitor.report(threshold=0.37)

if __name__ == "__main__":
    main()
    # ////////////////////
    #      _______
    #     /       \
    #    |  O   O  |
    #    |    ^    |
    #     \  ~~~  /
    #      \_____/
    #       _|_|_
    #      |     |
    #      |     | j. fuentes