import argparse, copy, glob, os, random, math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ECG_LEADS = ["i (mV)", "ii (mV)", "iii (mV)", "avr (mV)", "avl (mV)", "avf (mV)", "v1 (mV)", "v2 (mV)", "v3 (mV)", "v4 (mV)", "v5 (mV)", "v6 (mV)"]
LEAD_TO_INDEX = {name: idx for idx, name in enumerate(ECG_LEADS)}
DEFAULT_LEADS = ["ii (mV)", "v1 (mV)", "v5 (mV)"]
SEED = 42
OUTPUTDIR = "output"
DATAFOLDER = "VT-data"

# NEW: core timing defaults
SAMPLERATE = 32.0
HORIZON_SECONDS = 2 * 60.0
HORIZON_SAMPLES = int(HORIZON_SECONDS * SAMPLERATE)

# NEW: label and early-warning hyperparameters (easy to tweak)
# 1 - (d/horizon)^power
LABEL_RAMP_POWER = 2.0
# Strength of penalty for early high risk
EARLY_PENALTY_LAMBDA = 0.3
# Start penalising > 20 min before VT
EARLY_PENALTY_MIN = 20.0 * 60.0
# Target max probability far from VT
EARLY_PENALTY_MAX_PROB = 0.05

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_leads(names: Iterable[str]) -> List[int]:
    idx = []
    for name in names:
        key = name.strip().lower()
        matches = [k for k in LEAD_TO_INDEX if k.lower() == key]
        if not matches:
            raise ValueError(f"Unknown lead name '{name}'. Valid options: {ECG_LEADS}")
        idx.append(LEAD_TO_INDEX[matches[0]])
    return idx

def extract_patient_id(file_path: str) -> Optional[int]:
    base = os.path.basename(file_path)
    try:
        token = base.split("_")[1]
        return int(token.split(".")[0])
    except (IndexError, ValueError):
        return None

def available_patient_files(root: str = DATAFOLDER) -> List[str]:
    pattern = os.path.join(root, "Patient_*.h5")
    return sorted(glob.glob(pattern))

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@dataclass
class DatasetSummary:
    total_windows: int
    positive_windows: int
    negative_windows: int
    sample_rate: float
    window_seconds: float
    horizon_seconds: float
    patient_id: Optional[int] = None
    file_path: Optional[str] = None

class SinglePatientDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        channels: Optional[Iterable[int]] = None,
        sliding_window: int = 480,
        horizon: int = 320,
        step_size: Optional[int] = None,
        sample_rate: float = 32.0,
    ) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        self.file_path = file_path
        self.patient_id = extract_patient_id(file_path)
        self.sliding_window = int(sliding_window)
        self.horizon = int(horizon)
        self.sample_rate = float(sample_rate)
        self.step_size = int(step_size) if step_size is not None else max(1, self.sliding_window // 4)

        with h5py.File(self.file_path, "r") as f:
            data = f["data"]
            leads = data[:, 1:13]
            labels = data[:, 13].astype(np.int32)

        if channels is None:
            self.channel_idx = np.arange(leads.shape[1], dtype=int)
        else:
            self.channel_idx = np.array(channels, dtype=int)

        self.ecg = np.asarray(leads[:, self.channel_idx], dtype=np.float32)
        self.vt = labels

        total = self.ecg.shape[0]
        max_start = total - self.sliding_window - self.horizon
        if max_start <= 0:
            raise ValueError("Not enough samples for the chosen window/horizon.")

        # NEW: time-aware labels based on distance to next VT onset
        vt_bin = (self.vt == 1).astype(np.int32)
        N = vt_bin.shape[0]

        # NEW: precompute index of next VT sample at or after each position
        next_vt = np.full(N, -1, dtype=int)  # NEW
        last = -1  # NEW
        for i in range(N - 1, -1, -1):  # NEW
            if vt_bin[i] == 1:  # NEW
                last = i  # NEW
            next_vt[i] = last  # NEW

        labels_all = np.zeros(max_start + 1, dtype=np.float32)  # NEW
        time_to_vt_all = np.full(max_start + 1, np.inf, dtype=np.float32)  # NEW

        if self.horizon > 0:  # NEW
            horizon_f = float(self.horizon)  # NEW
            for start in range(max_start + 1):  # NEW
                # Consider the end of the current window as reference  # NEW
                idx = start + self.sliding_window  # NEW
                if idx >= N:  # NEW
                    continue  # NEW
                j = next_vt[idx]  # NEW
                if j < 0:  # no future VT  # NEW
                    continue  # NEW
                distance_samples = float(j - idx)  # NEW
                if distance_samples < 0:  # NEW
                    continue  # NEW
                distance_sec = distance_samples / self.sample_rate  # NEW
                time_to_vt_all[start] = distance_sec  # NEW
                if distance_samples <= self.horizon:  # within horizon  # NEW
                    # Linearly ramp label from 0 at horizon to 1 at onset  # NEW
                    labels_all[start] = 1.0 - distance_samples / horizon_f  # NEW
        else:  # NEW
            labels_all = np.zeros(max_start + 1, dtype=np.float32)  # NEW
            time_to_vt_all = np.full(max_start + 1, np.inf, dtype=np.float32)  # NEW

        starts = np.arange(0, max_start + 1, self.step_size, dtype=int)
        # NEW: treat any window with label > 0 as "positive", but weight by label strength
        pos_idx = np.where(labels_all > 0.0)[0]  # NEW
        if pos_idx.size > 0:  # NEW
            pad_base = max(self.horizon, self.step_size)  # NEW
            fine = max(1, self.step_size // 4)  # NEW
            extra = []  # NEW
            for pos in pos_idx:  # NEW
                strength = float(labels_all[pos])  # between 0 and 1  # NEW
                if strength <= 0.0:  # NEW
                    continue  # NEW
                # NEW: focus oversampling more strongly near high-label (near-onset) windows
                pad = max(self.step_size, int(pad_base * strength))  # NEW
                left = max(0, pos - pad)  # NEW
                right = int(pos)  # NEW
                extra.extend(range(left, right + 1, fine))  # NEW
            if extra:  # NEW
                starts = np.unique(  # NEW
                    np.clip(np.concatenate([starts, np.asarray(extra, dtype=int)]), 0, max_start)  # NEW
                )  # NEW

        self.starts = starts.astype(int)
        self.labels = labels_all[self.starts].astype(np.float32)  # NEW
        self.time_to_vt = time_to_vt_all[self.starts].astype(np.float32)  # NEW

        # NEW: treat labels > 0 as positive for summary
        pos_windows = int((self.labels > 0.0).sum())  # NEW
        neg_windows = int((self.labels <= 0.0).sum())  # NEW

        self.summary = DatasetSummary(
            total_windows=int(self.labels.size),
            positive_windows=pos_windows,  # NEW
            negative_windows=neg_windows,  # NEW
            sample_rate=self.sample_rate,
            window_seconds=float(self.sliding_window) / self.sample_rate,
            horizon_seconds=float(self.horizon) / self.sample_rate,
            patient_id=self.patient_id,
            file_path=self.file_path,
        )
        self.is_trivial = (
            self.summary.positive_windows == 0 or self.summary.negative_windows == 0
        )

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        segment = self.ecg[start : start + self.sliding_window]
        segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
        segment = segment - segment.mean(axis=0, keepdims=True)
        std = segment.std()
        if not np.isfinite(std) or std < 1e-3:
            std = 1e-3
        segment = segment / std
        x = torch.from_numpy(segment.astype(np.float32))
        y = torch.tensor([float(self.labels[idx])], dtype=torch.float32)
        # NEW: time-to-VT (seconds) for this window; inf if no future VT
        dist_sec = torch.tensor([float(self.time_to_vt[idx])], dtype=torch.float32)  # NEW
        return x, y, dist_sec  # NEW

class VTNETlite(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feat = self.encoder(x)
        return self.head(feat)

class PatientWindowCollection(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        drop_trivial: bool = True,
        **dataset_kwargs,
    ) -> None:
        self.patient_datasets: List[SinglePatientDataset] = []
        self.kept_files: List[str] = []
        self.kept_ids: List[int] = []
        self.summaries: List[DatasetSummary] = []
        self.trivial_summaries: List[DatasetSummary] = []
        self.failed: List[Tuple[str, str]] = []

        for path in files:
            try:
                patient_ds = SinglePatientDataset(path, **dataset_kwargs)
            except Exception as exc:
                self.failed.append((path, str(exc)))
                continue

            self.summaries.append(patient_ds.summary)
            if patient_ds.is_trivial:
                self.trivial_summaries.append(patient_ds.summary)
                if drop_trivial:
                    continue
            if len(patient_ds) == 0:
                self.trivial_summaries.append(patient_ds.summary)
                continue

            self.patient_datasets.append(patient_ds)
            self.kept_files.append(path)
            self.kept_ids.append(patient_ds.patient_id if patient_ds.patient_id is not None else -1)

        self.lengths = [len(ds) for ds in self.patient_datasets]
        self.cum_lengths = np.cumsum(self.lengths, dtype=int)
        self.labels = (
            np.concatenate([ds.labels for ds in self.patient_datasets])
            if self.lengths else np.array([], dtype=np.float32)
        )

    def __len__(self) -> int:
        return int(self.cum_lengths[-1]) if self.lengths else 0

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_idx = bisect_right(self.cum_lengths, idx)
        start = 0 if dataset_idx == 0 else self.cum_lengths[dataset_idx - 1]
        local_idx = idx - start
        return self.patient_datasets[dataset_idx][local_idx]

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def make_loader(self, batch_size: int, weighted: bool = True) -> DataLoader:
        if self.is_empty:
            raise ValueError("Dataset is empty; no patients available after filtering.")

        sampler = None
        shuffle = False
        labels = self.labels
        if weighted and labels.size > 0:
            # NEW: treat labels >= 0.5 as "positive" for sampling
            pos_mask = labels >= 0.5  # NEW
            neg_mask = labels < 0.5  # NEW
            pos = int(pos_mask.sum())  # NEW
            neg = int(neg_mask.sum())  # NEW
            if pos > 0 and neg > 0:  # NEW
                weights = np.where(pos_mask, 1.0 / max(pos, 1), 1.0 / max(neg, 1))  # NEW
                sampler = WeightedRandomSampler(  # NEW
                    weights=weights.tolist(), num_samples=len(labels), replacement=True  # NEW
                )  # NEW
            else:  # NEW
                shuffle = True  # NEW
        elif not weighted:
            shuffle = False
        else:
            shuffle = True

        return DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=shuffle, drop_last=False)

def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    batches = 0

    # # NEW: hyperparameters for early high-probability penalty
    # EARLY_PENALTY_LAMBDA = 0.1
    # # Penalise very early high risk
    # EARLY_PENALTY_MIN = 20.0 * 60.0
    # # Encourage low prob far from VT
    # EARLY_PENALTY_MAX_PROB = 0.1

    for batch in loader:
        if len(batch) == 3:  # NEW
            x, y, dist = batch  # NEW
        else:  # NEW
            x, y = batch  # NEW
            dist = None  # NEW

        x = x.to(device)
        y = y.to(device)
        dist = dist.to(device) if dist is not None else None  # NEW

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        base_loss = criterion(logits, y)
        loss = base_loss  # NEW

        # NEW: add penalty for very early windows with high probability
        if dist is not None and EARLY_PENALTY_LAMBDA > 0.0:  # NEW
            probs = torch.sigmoid(logits)  # NEW
            # Windows far from VT (or with no VT: dist ~ inf)  # NEW
            early_mask = dist.view_as(probs) > EARLY_PENALTY_MIN  # NEW
            if early_mask.any():  # NEW
                early_probs = probs[early_mask]  # NEW
                if early_probs.numel() > 0:  # NEW
                    penalty = ((early_probs - EARLY_PENALTY_MAX_PROB).clamp(min=0.0) ** 2).mean()  # NEW
                    loss = loss + EARLY_PENALTY_LAMBDA * penalty  # NEW

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches)

def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:  # NEW
                x, y, _ = batch  # NEW
            else:  # NEW
                x, y = batch  # NEW
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
    if probs:
        probs_arr = np.vstack(probs).flatten()
        labels_arr = np.vstack(labels).flatten()
    else:
        probs_arr = np.array([], dtype=float)
        labels_arr = np.array([], dtype=float)
    return float(np.mean(losses)) if losses else float("nan"), probs_arr, labels_arr

def summary(probs: np.ndarray, labels: np.ndarray) -> dict:
    if probs.size == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": float("nan")}
    
    # NEW: treat labels >= 0.5 as positive when computing metrics
    bin_labels = (labels >= 0.5).astype(int)  # NEW

    grid = np.linspace(0.1, 0.9, 17)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    for thr in grid:
        preds = (probs >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(bin_labels, preds, average="binary", zero_division=0)  # NEW
        if f1 > best["f1"]:
            best.update({"threshold": float(thr), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    try:
        roc_auc = float(roc_auc_score(bin_labels, probs)) if len(np.unique(bin_labels)) > 1 else float("nan")  # NEW
    except Exception:
        roc_auc = float("nan")

    best["roc_auc"] = roc_auc

    return best

def smooth_probs(probs: np.ndarray, time_seconds: np.ndarray, window: int = 0, half_life: float = 0.0) -> np.ndarray:
    if probs.size == 0:
        return probs

    smoothed = np.asarray(probs, dtype=float).copy()

    if window is not None and int(window) > 1:
        kernel = np.ones(int(window), dtype=float) / float(window)
        smoothed = np.convolve(smoothed, kernel, mode="same")

    if half_life is not None and float(half_life) > 0.0 and smoothed.size > 1:
        tau = float(half_life) / math.log(2.0)
        ema = np.empty_like(smoothed, dtype=float)
        ema[0] = smoothed[0]
        for i in range(1, smoothed.size):
            dt = max(1e-6, float(time_seconds[i] - time_seconds[i - 1]))
            alpha = 1.0 - math.exp(-dt / tau)
            ema[i] = (1.0 - alpha) * ema[i - 1] + alpha * smoothed[i]
        smoothed = ema

    return smoothed

def plot_ew(
    dataset: SinglePatientDataset,
    probs: np.ndarray,
    threshold: float,
    output_path: str,
    smooth_window: int = 0,
    smooth_half_life: float = 0.0,
    pre_smoothed: Optional[np.ndarray] = None,
    display_alarm_time: Optional[float] = None,
    max_lead_time: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    
    if probs.size == 0:
        return None, None, None

    time_seconds = dataset.starts.astype(float) / dataset.sample_rate
    time_minutes = time_seconds / 60.0
    horizon_labels = dataset.labels.astype(float)
    vt_indices = np.where(dataset.vt == 1)[0]
    onset_time = float(vt_indices[0] / dataset.sample_rate) if vt_indices.size > 0 else None

    if pre_smoothed is not None:
        smooth = np.asarray(pre_smoothed, dtype=float)
        if smooth.shape != probs.shape:
            raise ValueError("pre_smoothed probabilities must match probs shape")
    else:
        smooth = smooth_probs(probs, time_seconds, smooth_window, smooth_half_life)
    smoothing_active = pre_smoothed is not None or smooth_window > 1 or smooth_half_life > 0.0
    crosses = np.where(smooth >= float(threshold))[0]
    first_cross = float(time_seconds[crosses[0]]) if crosses.size > 0 else None
    lead_time = None

    if onset_time is not None and first_cross is not None:
        lead_time = float(max(0.0, onset_time - first_cross))

    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    ax.step(time_minutes, horizon_labels, where="post", color="#e83875", alpha=0.6, linewidth=2.0, label="Future VT label")

    #if smoothing_active:
    #    ax.plot(time_minutes, probs, color="#a2adad", linewidth=1.2, alpha=0.6, label="Probability (raw)")

    prob_label = "Probability" if smoothing_active else "Predicted probability"
    ax.plot(time_minutes, smooth, color="#4aa5e2", linewidth=2.0, label=prob_label)
    ax.axhline(threshold, color="#555555", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.2f}")

    if onset_time is not None:
        ax.axvline(onset_time / 60.0, color="#d36aa2", linestyle="--", linewidth=2.0, label=f"VT onset ({onset_time/60:.1f} min)")

    if max_lead_time is not None and max_lead_time > 0.0 and onset_time is not None:
        window_start = max(0.0, onset_time - max_lead_time)
        ax.axvspan(
            window_start / 60.0,
            onset_time / 60.0,
            color="#C8F05B",
            alpha=0.25,
            label=f"Window (<={max_lead_time/60:.1f} min)",
        )
    alarm_time = display_alarm_time if display_alarm_time is not None else first_cross

    if alarm_time is not None:
        label = "Alarm"
        if display_alarm_time is not None and (first_cross is None or not math.isclose(alarm_time, first_cross, rel_tol=1e-6, abs_tol=1e-3)):
            label = "Accepted alarm"
        label = f"{label} ({alarm_time/60:.1f} min)"
        ax.axvline(alarm_time / 60.0, color="#C089D9", linestyle="-", linewidth=2.0, label=label)
        ax.scatter([alarm_time / 60.0], [threshold], color="#C089D9", s=60, zorder=5)

    def timeform(x, _):
        total_seconds = float(x) * 60.0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    ax.xaxis.set_major_formatter(FuncFormatter(timeform))
    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Probability / Label")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return first_cross, lead_time, onset_time

def run_training(args: argparse.Namespace) -> None:
    set_seed(SEED)

    lead_names = list(DEFAULT_LEADS)
    channels = parse_leads(lead_names)

    all_files = available_patient_files(args.data_root)
    if not all_files:
        raise FileNotFoundError(f"No patient files found under '{args.data_root}'.")

    mapping = {}
    ids = []
    for path in all_files:
        pid = extract_patient_id(path)
        if pid is None:
            continue
        mapping[pid] = path
        ids.append(pid)
    ids.sort()
    if not ids:
        raise RuntimeError("Could not extract patient IDs from filenames.")

    def parse_ids(text: Optional[str], default: List[int]) -> List[int]:
        if text is None:
            return list(default)
        seen = set()
        result = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                pid = int(token)
            except ValueError:
                raise ValueError(f"Invalid patient id '{token}'. Use integers like 1,2,3.")
            if pid not in mapping:
                raise FileNotFoundError(f"Patient {pid} not found in data root.")
            if pid not in seen:
                seen.add(pid)
                result.append(pid)
        return sorted(result)

    eval_default = ids[-max(1, min(args.eval_count, len(ids))):]
    eval_ids = parse_ids(args.eval_patients, eval_default)
    train_default = [pid for pid in ids if pid not in eval_ids]
    train_ids = parse_ids(args.train_patients, train_default)
    if not train_ids:
        raise ValueError("Training patient set is empty after removing overlaps with evaluation patients.")

    train_files = [mapping[pid] for pid in train_ids]
    eval_files = [mapping[pid] for pid in eval_ids]

    print(f"Training patients: {[extract_patient_id(f) for f in train_files]}")
    print(f"Evaluation patients: {[extract_patient_id(f) for f in eval_files]}")

    dataset_kwargs = dict(
        channels=channels,
        sliding_window=int(args.sliding_window),
        horizon=int(args.horizon),
        step_size=args.step_size,
        sample_rate=args.sample_rate,
    )

    train_dataset = PatientWindowCollection(train_files, drop_trivial=True, **dataset_kwargs)
    eval_dataset = PatientWindowCollection(eval_files, drop_trivial=True, **dataset_kwargs)

    if train_dataset.trivial_summaries:
        skipped = [s.patient_id for s in train_dataset.trivial_summaries]
        print(f"Skipped {len(skipped)} trivial training patients: {skipped}")
    if train_dataset.failed:
        print("Failed to load training patients:")
        for path, err in train_dataset.failed:
            print(f"  {path}: {err}")

    if train_dataset.is_empty:
        raise RuntimeError("No training data available after filtering trivial or failed patients.")

    if eval_dataset.failed:
        print("Failed to load evaluation patients:")
        for path, err in eval_dataset.failed:
            print(f"  {path}: {err}")
    if eval_dataset.trivial_summaries:
        info = [s.patient_id for s in eval_dataset.trivial_summaries]
        print(f"Evaluation set contains {len(info)} trivial patients (excluded from evaluation metrics): {info}")

    train_loader = train_dataset.make_loader(args.batch_size, weighted=True)
    eval_loader = eval_dataset.make_loader(args.batch_size, weighted=False) if not eval_dataset.is_empty else None

    device = pick_device()
    model = VTNETlite(n_channels=len(channels)).to(device)

    train_labels = train_dataset.labels
    if train_labels.size > 0:
        # NEW: interpret labels >= 0.5 as positive for reporting
        positives = int((train_labels >= 0.5).sum())  # NEW
        negatives = int((train_labels < 0.5).sum())  # NEW
        print(f"Training windows: total={len(train_labels)} positives={positives} negatives={negatives}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        if eval_loader is not None:
            val_loss, val_probs, val_labels = evaluate(model, eval_loader, criterion, device)
            metrics = summary(val_probs, val_labels)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_f1={metrics['f1']:.3f} thr={metrics['threshold']:.2f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} (no evaluation dataset)")

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = OUTPUTDIR
    os.makedirs(output_dir, exist_ok=True)

    aggregate_metrics = None
    aggregate_probs = np.array([], dtype=float)
    aggregate_labels = np.array([], dtype=float)
    aggregate_loss = float("nan")
    
    if eval_loader is not None:
        aggregate_loss, aggregate_probs, aggregate_labels = evaluate(model, eval_loader, criterion, device)
        aggregate_metrics = summary(aggregate_probs, aggregate_labels)
        print("\nEvaluation (aggregate across held-out patients)")
        print(
            f"loss={aggregate_loss:.4f}  F1={aggregate_metrics['f1']:.3f}  "
            f"precision={aggregate_metrics['precision']:.3f}  recall={aggregate_metrics['recall']:.3f}  "
            f"threshold={aggregate_metrics['threshold']:.2f}  ROC-AUC={aggregate_metrics['roc_auc']:.3f}"
        )
        # NEW: initialise accumulators for smoothed, patient-specific metrics
        smoothed_all_probs = []  # NEW
        smoothed_all_labels = []  # NEW
        smoothed_tp = smoothed_fp = smoothed_tn = smoothed_fn = 0  # NEW
        smoothed_roc_auc = float("nan")  # NEW
        smoothed_precision = float("nan")  # NEW
        smoothed_recall = float("nan")  # NEW
        smoothed_f1 = float("nan")  # NEW
    else:
        print("\nNo evaluation dataset available; skipping aggregate metrics.")

    patient_reports = {}
    global_threshold = aggregate_metrics["threshold"] if aggregate_metrics is not None else 0.5

    for patient_ds in eval_dataset.patient_datasets:
        loader = DataLoader(patient_ds, batch_size=args.batch_size, shuffle=False)
        loss, probs, labels = evaluate(model, loader, criterion, device)

        # Raw (unsmoothed) per-patient metrics  # NEW
        per_metrics_raw = summary(probs, labels)  # NEW
        preds_global = (probs >= global_threshold).astype(int) if probs.size else np.array([], dtype=int)
        if labels.size:  # NEW
            labels_bin = (labels >= 0.5).astype(int)  # NEW
            precision_g, recall_g, f1_g, _ = precision_recall_fscore_support(  # NEW
                labels_bin, preds_global, average="binary", zero_division=0  # NEW
            )  # NEW
        else:  # NEW
            labels_bin = np.array([], dtype=int)  # NEW
            precision_g, recall_g, f1_g, _ = 0.0, 0.0, 0.0, None  # NEW

        # Smoothed probabilities and patient-specific threshold  # NEW
        time_seconds = patient_ds.starts.astype(float) / patient_ds.sample_rate  # NEW
        smoothed_probs = smooth_probs(  # NEW
            probs, time_seconds, args.smooth_window, args.smooth_half_life  # NEW
        )  # NEW
        if eval_loader is not None:  # NEW
            smoothed_all_probs.append(smoothed_probs)  # NEW
            smoothed_all_labels.append(labels)  # NEW

        per_metrics_smooth = summary(smoothed_probs, labels)  # NEW
        patient_threshold = per_metrics_smooth["threshold"]  # NEW
        preds_patient = (smoothed_probs >= patient_threshold).astype(int) if smoothed_probs.size else np.array([], dtype=int)  # NEW

        if labels_bin.size:  # NEW
            # confusion components for this patient under smoothed, patient-specific threshold  # NEW
            tp = int(((labels_bin == 1) & (preds_patient == 1)).sum())  # NEW
            fp = int(((labels_bin == 0) & (preds_patient == 1)).sum())  # NEW
            tn = int(((labels_bin == 0) & (preds_patient == 0)).sum())  # NEW
            fn = int(((labels_bin == 1) & (preds_patient == 0)).sum())  # NEW
            if eval_loader is not None:  # NEW
                smoothed_tp += tp  # NEW
                smoothed_fp += fp  # NEW
                smoothed_tn += tn  # NEW
                smoothed_fn += fn  # NEW

            # per-patient F1 at patient-specific smoothed threshold  # NEW
            prec_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # NEW
            rec_p = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # NEW
            f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0  # NEW
        else:  # NEW
            prec_p = rec_p = f1_p = 0.0  # NEW

        patient_id = patient_ds.patient_id if patient_ds.patient_id is not None else -1
        plot_path = os.path.join(output_dir, f"beatbot_patient_{patient_id}.png")
        # Use the patient-specific threshold for the early-warning plot  # NEW
        first_alarm, lead_time, onset_time = plot_ew(
            patient_ds,
            probs,
            patient_threshold,  # NEW
            plot_path,
            smooth_window=args.smooth_window,
            smooth_half_life=args.smooth_half_life,
        )

        print(
            f"Patient {patient_id}: loss={loss:.4f}  "
            f"F1(global_raw)={f1_g:.3f}  precision_global_raw={precision_g:.3f}  recall_global_raw={recall_g:.3f}  "
            f"best_f1_raw={per_metrics_raw['f1']:.3f}  "
            f"F1(patient_smoothed)={f1_p:.3f}"
        )  # NEW

        patient_reports[str(patient_id)] = {
            "loss": float(loss),
            "best_threshold_raw": per_metrics_raw["threshold"],  # NEW
            "best_f1_raw": per_metrics_raw["f1"],  # NEW
            "roc_auc_raw": per_metrics_raw["roc_auc"],  # NEW
            "patient_threshold_smoothed": float(patient_threshold),  # NEW
            "best_f1_smoothed": float(per_metrics_smooth["f1"]),  # NEW
            "roc_auc_smoothed": float(per_metrics_smooth["roc_auc"]),  # NEW
            "precision_global_raw": float(precision_g),  # NEW
            "recall_global_raw": float(recall_g),  # NEW
            "f1_global_raw": float(f1_g),  # NEW
            "f1_patient_smoothed": float(f1_p),  # NEW
            "first_alarm_seconds": first_alarm,
            "lead_time_seconds": lead_time,
            "onset_time_seconds": onset_time,
            "plot": plot_path,
        }

    # NEW: aggregate smoothed ROC-AUC (pooled windows) and global metrics under patient-specific thresholds
    if eval_loader is not None:
        if smoothed_all_probs:
            agg_smoothed_probs = np.concatenate(smoothed_all_probs)  # NEW
            agg_smoothed_labels = np.concatenate(smoothed_all_labels)  # NEW
            smoothed_summary = summary(agg_smoothed_probs, agg_smoothed_labels)  # NEW
            smoothed_roc_auc = smoothed_summary["roc_auc"]  # NEW

        # NEW: derive global precision/recall/F1 from accumulated confusion matrix
        total_tp = smoothed_tp  # NEW
        total_fp = smoothed_fp  # NEW
        total_tn = smoothed_tn  # NEW
        total_fn = smoothed_fn  # NEW

        if (total_tp + total_fp) > 0:  # NEW
            smoothed_precision = total_tp / (total_tp + total_fp)  # NEW
        else:  # NEW
            smoothed_precision = 0.0  # NEW
        if (total_tp + total_fn) > 0:  # NEW
            smoothed_recall = total_tp / (total_tp + total_fn)  # NEW
        else:  # NEW
            smoothed_recall = 0.0  # NEW
        if (smoothed_precision + smoothed_recall) > 0:  # NEW
            smoothed_f1 = 2 * smoothed_precision * smoothed_recall / (smoothed_precision + smoothed_recall)  # NEW
        else:  # NEW
            smoothed_f1 = 0.0  # NEW

        print("\nEvaluation (smoothed, patient-specific thresholds)")  # NEW
        print(
            f"F1={smoothed_f1:.3f}  precision={smoothed_precision:.3f}  "
            f"recall={smoothed_recall:.3f}  ROC-AUC(smoothed)={smoothed_roc_auc:.3f}"
        )  # NEW

    model_path = os.path.join(output_dir, "beatbot_weights.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved weights to {model_path}")

    if train_dataset.patient_datasets:
        inferred_step = train_dataset.patient_datasets[0].step_size
    elif eval_dataset.patient_datasets:
        inferred_step = eval_dataset.patient_datasets[0].step_size
    else:
        inferred_step = args.step_size

    meta = {
        "data_root": args.data_root,
        "train_patients": [extract_patient_id(f) for f in train_dataset.kept_files],
        "eval_patients": [extract_patient_id(f) for f in eval_dataset.kept_files],
        "skipped_train": [s.patient_id for s in train_dataset.trivial_summaries],
        "failed_train": train_dataset.failed,
        "failed_eval": eval_dataset.failed,
        "smoothing": {
            "window": int(args.smooth_window),
            "half_life": float(args.smooth_half_life),
        },
        "sliding_window": int(args.sliding_window),
        "horizon": int(args.horizon),
        "step_size": inferred_step,
        "sample_rate": args.sample_rate,
        "leads": lead_names,
        "aggregate": {
            "loss": aggregate_loss,
            "metrics": aggregate_metrics if aggregate_metrics is not None else None,
            "metrics_raw": aggregate_metrics if aggregate_metrics is not None else None,  # NEW
            "metrics_smoothed_patient": {  # NEW
                "precision": float(smoothed_precision),  # NEW
                "recall": float(smoothed_recall),  # NEW
                "f1": float(smoothed_f1),  # NEW
                "roc_auc_smoothed": float(smoothed_roc_auc),  # NEW
                "threshold_strategy": "patient_specific",  # NEW
                "smoothing": {  # NEW
                    "window": int(args.smooth_window),  # NEW
                    "half_life": float(args.smooth_half_life),  # NEW
                },  # NEW
            } if eval_loader is not None else None,  # NEW
        },
        "patients": patient_reports,
    }
    meta_path = os.path.join(output_dir, "beatbot_meta.json")
    with open(meta_path, "w") as f:
        import json

        json.dump(meta, f, indent=2, default=float)
    print(f"Saved metadata to {meta_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate VTNETlite model")
    parser.add_argument("--data-root", default=DATAFOLDER, help="Directory containing Patient_*.h5 files")
    parser.add_argument("--train-patients", help="Comma-separated patient IDs to train on")
    parser.add_argument("--eval-patients", help="Comma-separated patient IDs to evaluate on")
    parser.add_argument("--eval-count", type=int, default=2, help="Default number of patients reserved for evaluation")

    parser.add_argument("--sliding-window", type=int, default=int(15.0 * SAMPLERATE), help="Window length in samples")
    parser.add_argument("--horizon", type=int, default=HORIZON_SAMPLES, help="Positive label horizon in samples")
    parser.add_argument("--step-size", type=int, default=None, help="Step between consecutive windows")
    parser.add_argument("--sample-rate", type=float, default=SAMPLERATE, help="Sampling frequency in Hz")

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--smooth-window", type=int, default=10, help="Simple moving-average window (in windows)")
    parser.add_argument("--smooth-half-life", type=float, default=10.0, help="Exponential smoothing half-life in sec")
    return parser

# ----------------------------------------------------------------------------------/
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)

if __name__ == "__main__":
    main()