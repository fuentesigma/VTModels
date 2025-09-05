import os
import glob
import h5py
import numpy as np
# //////////////////////////////////////////////////////////////////////////
import warnings
import multiprocessing
# //////////////////////////////////////////////////////////////////////////
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
# //////////////////////////////////////////////////////////////////////////
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, get_worker_info
# //////////////////////////////////////////////////////////////////////////
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
# //////////////////////////////////////////////////////////////////////////
"""
    * P A T I E N T S *
"""
# Gather all existing patient HDF5 files, excluding outlier files
pattern = os.path.join("VT-data", "Patient_*.h5")
allfiles = glob.glob(pattern)

# Exclude outlier files suffixed as '_o.h5'
files = [f for f in allfiles if not os.path.basename(f).endswith('_o.h5')]
numbers = [os.path.basename(f) for f in files]

# Extract patient IDs from filenames
cohort = sorted([int(fname.split("_")[1].split(".")[0]) for fname in numbers])

# Reserve one patient (last in the sorted list) for final evaluation
evaluation_idx = [cohort[-1]]

# Use remaining patients for train/validation split
remaining = [pid for pid in cohort if pid not in evaluation_idx]

# Split 20% of remaining into validation
train_idx, validation_idx = train_test_split(remaining, test_size=0.2, random_state=42)

# File lists based on splits
trainCohort = [f"VT-data/Patient_{i}.h5" for i in train_idx]
validationCohort = [f"VT-data/Patient_{i}.h5" for i in validation_idx]
evaluationCohort = [f"VT-data/Patient_{i}.h5" for i in evaluation_idx]

# //////////////////////////////////////////////////////////////////////////
# Fixed parameters
SLIDING_WINDOW = int(15 * 32)

input_dim = 12
num_epochs = 100    # Number of epochs for final training
ssl_epochs = 20     # Number of epochs for pre-training
wnb_epochs = 20     # Number of epochs for fine-tuning

# ---------------------------------------------------------------------------------/
class BeatHarvest(Dataset):
    def __init__(self, files, sliding_window=320, horizon=320, batch_size=16, overlap=0.25, num_workers=4):
        self.files = files
        self.sliding_window = sliding_window
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers

        if not (0 <= overlap < 1):
            raise ValueError("Overlap must lie in [0,1).")
        self.step_size = int(sliding_window * (1 - overlap))

        if self.step_size <= 0:
            raise ValueError("Step size must be positive.")

        self._h5_files = {}
        self.file_metas = []
        self.indices = []
        self.sample_labels = []

        self.loader()
        self.atlas()

        # Use the sampler
        self.sampler = self.wsampler()

    def push(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn
        )

    def loader(self):
        for file_path in self.files:
            with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
                length = f['data'].shape[0]
            self.file_metas.append({'file_path': file_path, 'length': length})

    def atlas(self):
        for fidx, meta in enumerate(self.file_metas):

            length = meta['length']
            max_start = length - self.sliding_window - self.horizon
            
            if max_start < 0:
                continue
            
            for start in range(0, max_start + 1, self.step_size):

                with h5py.File(meta['file_path'], 'r', libver='latest', swmr=True) as f:
                    data = f['data'][:]
                    vt_future = data[start + self.sliding_window : start + self.sliding_window + self.horizon, 13]
                
                label = int(np.any(vt_future == 1))

                self.indices.append((fidx, start))
                self.sample_labels.append(label)

        if len(self.indices) == 0:
            raise ValueError("No valid windows were generated.")

    def wsampler(self):
        labels = np.array(self.sample_labels)
        count_pos = np.sum(labels == 1)
        count_neg = np.sum(labels == 0)
        
        if count_pos == 0 or count_neg == 0:
            warnings.warn(f"Patient file {self.files} has trivial classes. Skipping.")
            return None
        
        w_pos = 1.0 / (count_pos + 1e-6)
        w_neg = 1.0 / (count_neg + 1e-6)
        weights = np.where(labels == 1, w_pos, w_neg)
        
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def __getitem__(self, idx):
        fidx, start = self.indices[idx]

        if fidx not in self._h5_files:
            file_path = self.file_metas[fidx]['file_path']
            self._h5_files[fidx] = h5py.File(file_path, 'r', libver='latest', swmr=True)

        f = self._h5_files[fidx]
        data = f['data']
        ecg = data[start : start + self.sliding_window, 1:13]

        x = torch.as_tensor(ecg, dtype=torch.float32)
        y = torch.as_tensor([self.sample_labels[idx]], dtype=torch.float32)
        
        return x, y

    def __len__(self):
        return len(self.indices)
    
    def __del__(self):
        if hasattr(self, '_h5_files'):
            for f in self._h5_files.values():
                try:
                    f.close()
                except Exception:
                    pass

    def _worker_init_fn(self, worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        dataset._h5_files = {}
    
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # Use a padding to ensure the output size matches the input size
        padding = (kernel_size - 1) * dilation

        # First convolution layer with weight normalisation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_channels)

        # Second convolution layer with weight normalisation
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Downsample layer: only if in_channels != out_channels
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None)

    def forward(self, x):
        # Recall the input has shape (batch, channels, seq_len)
        seq_len = x.size(2)

        # First conv + norm + activation + dropout
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv + norm + activation + dropout
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Crop to original length and residual
        out = out[:, :, :seq_len]
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class VTATTEND(nn.Module):
    def __init__(self, input_dim, channel_sizes, kernel_size, num_heads, dropout, horizon):
        super().__init__()

        # Temporal convolutional encoder
        layers = []
        for i, out_ch in enumerate(channel_sizes):
            in_ch = input_dim if i == 0 else channel_sizes[i-1]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        # Temporal Convolutional Network
        self.tcn = nn.Sequential(*layers)

        # Self-attention decoder
        self.attention = nn.MultiheadAttention(embed_dim=channel_sizes[-1], num_heads=num_heads, dropout=dropout, batch_first=True)

        # Prediction head
        self.fc = nn.Linear(channel_sizes[-1], 1)

    def forward(self, x, return_sequence: bool = False):
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Temporal convolutional encoder
        y = self.tcn(x)
        
        # Back to (batch, seq_len, channels)
        y = y.transpose(1, 2)
        
        # Self-attention decoder
        attn_out, _ = self.attention(y, y, y)

        if return_sequence:
            return attn_out
        
        # Use only the final time step for classification
        z = attn_out[:, -1, :]
        
        # Prediction head
        return self.fc(z)

class Cardiobot:
    def __init__(self, model, train_loader, val_loader=None, lr=1e-4, weight_decay=1e-5):
        # Perform device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False

        # Self-supervised components
        self.mask_ratio = 0.3
        # model.attention.embed_dim holds the sequence embedding dimension
        d_model = self.model.attention.embed_dim
        # Decoder: project from sequence embeddings back to 12-lead ECG
        self.pre_decoder = nn.Linear(d_model, 12).to(self.device)
        # MSE loss for reconstruction
        self.pre_criterion = nn.MSELoss()
        # Optimiser for encoder+decoder during SSL
        self.pre_optimiser = optim.Adam(
            list(self.model.parameters()) + list(self.pre_decoder.parameters()),
            lr=1e-4
        )

    def train_epoch(self, epoch_index):
        """
        Supervised training for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for ecg, labels in self.train_loader:
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(ecg)
            loss = self.criterion(logits, labels)

            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(1, num_batches)

        # Update learning rate
        self.scheduler.step()

        # ////////////////////////////////////////////////////////////////////////////////////
        print(f"Epoch {epoch_index + 1}/{self.num_epochs} Training Loss: {epoch_loss:.6f}")
        # ////////////////////////////////////////////////////////////////////////////////////
        return epoch_loss

    def validate_epoch(self, epoch_index):
        """
        Supervised validation for one epoch.
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        true_list = []
        pred_list = []

        with torch.no_grad():
            for ecg, labels in self.val_loader:
                ecg = ecg.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(ecg)
                loss = self.criterion(logits, labels)

                # Convert logits to probabilities and threshold for binary prediction
                probs = torch.sigmoid(logits).view(-1)
                preds = (probs > 0.5).long()
                pred_list.extend(preds.cpu().tolist())
                true_list.extend(labels.cpu().view(-1).tolist())

                running_loss += loss.item()
                num_batches += 1

        epoch_loss = running_loss / max(1, num_batches)

        # ////////////////////////////////////////////////////////////////////////////////////
        print(f"Epoch {epoch_index + 1}/{self.num_epochs} Validation Loss: {epoch_loss:.6f}")
        # ////////////////////////////////////////////////////////////////////////////////////

        # Compute precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(true_list, pred_list, average='binary', zero_division=0)
        return epoch_loss

    def pretrain_epoch(self, epoch_index):
        """
        Self-supervised pretraining over masked windows.
        """
        self.model.train()
        self.pre_decoder.train()
        total_loss = 0.0

        for ecg, _ in self.train_loader:
            # Move to device
            ecg = ecg.to(self.device)

            # Reshape ECG to (batch, seq_len, channels)
            batch, seq_len, channels = ecg.size()

            # Randomly mask a fraction of the sequence
            mask = (torch.rand(batch, seq_len, device=self.device) < self.mask_ratio)

            # Expand mask to match the ECG shape and clone it
            mask_expanded = mask.unsqueeze(-1).expand_as(ecg)
            masked_ecg = ecg.clone()

            # Set masked positions to zero
            masked_ecg[mask_expanded] = 0.0

            # Obtain sequence embeddings (requires model to support return_sequence=True)
            seq_embed = self.model(masked_ecg, return_sequence=True)

            # Decode back
            recon = self.pre_decoder(seq_embed)
            # recon is (batch, seq_len, channels)

            # Compute MSE on masked positions only
            loss = self.pre_criterion(recon[mask_expanded], ecg[mask_expanded])

            # Typical optimisation steps
            self.pre_optimiser.zero_grad()
            loss.backward()
            self.pre_optimiser.step()

            # Accumulate loss
            total_loss += loss.item()

        # Average loss over the epoch
        avg_loss = total_loss / max(1, len(self.train_loader))

        # ////////////////////////////////////////////////////////////////////////////////////
        print(f"Pretrain Epoch {epoch_index+1}, Loss: {avg_loss:.6f}")
        # ////////////////////////////////////////////////////////////////////////////////////

        return avg_loss

    def fit(self, num_epochs, pre_epochs, patience, ssl=False):
        self.num_epochs = num_epochs
        self.pre_epochs = pre_epochs

        # Scheduler for supervised phase
        if num_epochs > 0:
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=self.num_epochs, eta_min=1e-6)

        # ////////////////////////////////////////////////////////////////////////////////////
        if ssl and self.pre_epochs > 0:
            for epoch in range(self.pre_epochs):
                _ = self.pretrain_epoch(epoch)

        # ////////////////////////////////////////////////////////////////////////////////////
        best_val_loss = float('inf')
        lazy_epochs = 0

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                lazy_epochs = 0
            else:
                lazy_epochs += 1

            if lazy_epochs >= patience:
                # //////////////////////////////////////////////////////////////
                print(f"Early stopping at epoch {epoch + 1}.")
                # //////////////////////////////////////////////////////////////
                break

# ----------------------------------------------------------------------------------/
class Pipeline:
    def __init__(self):
        # Initialise grid for experiment tracking with sensible defaults
        self.config = {
            "channel_sizes": [[16, 32, 64], [32, 64, 128], [64, 128, 256]],
            "kernel_size": 3,
            "num_heads": 4,
            "dropout": 0.6,
            "horizon": 320,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
        }

    def objective(self, trial):
        # Suggest hyperparameters
        _channel_sizes_opts = ((16, 32, 64), (32, 64, 128), (64, 128, 256))
        _cs_ix = trial.suggest_categorical("channel_sizes_ix", list(range(len(_channel_sizes_opts))))
        channel_sizes = list(_channel_sizes_opts[_cs_ix])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.3, 0.7)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        horizon = trial.suggest_categorical("horizon", [160, 320, 480])
        patience = trial.suggest_categorical("patience", [2, 3, 5])

        files = [f"VT-data/Patient_{pid}.h5" for pid in remaining]
        groups = remaining

        kf = GroupKFold(n_splits=5)
        validation_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(files, groups=groups)):
            print(f"Starting cross-validation fold {fold + 1}/{kf.n_splits}")
            train_files = [files[i] for i in train_idx]
            validation_files = [files[i] for i in val_idx]

            train_loader = BeatHarvest(
                train_files,
                SLIDING_WINDOW,
                horizon,
                batch_size
            ).push()

            validation_loader = BeatHarvest(
                validation_files,
                SLIDING_WINDOW,
                horizon,
                batch_size
            ).push()

            model = VTATTEND(
                input_dim=input_dim,
                channel_sizes=channel_sizes,
                kernel_size=kernel_size,
                num_heads=num_heads,
                dropout=dropout,
                horizon=horizon
            )

            trainer = Cardiobot(
                model,
                train_loader,
                validation_loader,
                lr=learning_rate,
                weight_decay=weight_decay
            )

            # Train with early stopping
            trainer.fit(num_epochs=wnb_epochs, pre_epochs=0, patience=patience, ssl=False)

            # Evaluate on the current validation split
            validation_loss = trainer.validate_epoch(0)
            validation_losses.append(validation_loss)

            # Intermediate report to enable pruning
            trial.report(validation_loss, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        avg_val_loss = sum(validation_losses) / len(validation_losses)
        return avg_val_loss

    def fulltrain(self, best_cfg):
        train_loader = BeatHarvest(trainCohort, SLIDING_WINDOW, best_cfg["horizon"], best_cfg["batch_size"]).push()
        validation_loader = BeatHarvest(validationCohort, SLIDING_WINDOW, best_cfg["horizon"], best_cfg["batch_size"]).push()

        model = VTATTEND(
            input_dim=input_dim,
            channel_sizes=best_cfg["channel_sizes"],
            kernel_size=best_cfg["kernel_size"],
            num_heads=best_cfg["num_heads"],
            dropout=best_cfg["dropout"],
            horizon=best_cfg["horizon"]
        )

        trainer = Cardiobot(
            model,
            train_loader=train_loader,
            val_loader=validation_loader,
            lr=best_cfg["learning_rate"],
            weight_decay=best_cfg["weight_decay"]
        )

        trainer.fit(num_epochs=num_epochs, pre_epochs=ssl_epochs, patience=num_epochs//4, ssl=True)

        torch.save(model.state_dict(), "best_VTATTEND.pth")
        print("Final model saved as 'best_VTATTEND.pth'")

    def controlpanel(self):
        multiprocessing.freeze_support()
        if multiprocessing.current_process().name == "MainProcess":
            n_trials = int(os.getenv("OPTUNA_TRIALS", "30"))
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
            study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
            study.optimize(self.objective, n_trials=n_trials, show_progress_bar=False)

            print("Best hyperparameters found:")
            print(study.best_trial.params)

            # Reconstruct channel_sizes from stored index
            _channel_sizes_opts = ((16, 32, 64), (32, 64, 128), (64, 128, 256))
            _best_cs = list(_channel_sizes_opts[study.best_trial.params["channel_sizes_ix"]])

            # Build best config for final training
            best_cfg = {
                "channel_sizes": _best_cs,
                "kernel_size": study.best_trial.params["kernel_size"],
                "num_heads": study.best_trial.params["num_heads"],
                "dropout": study.best_trial.params["dropout"],
                "horizon": study.best_trial.params["horizon"],
                "batch_size": study.best_trial.params["batch_size"],
                "learning_rate": study.best_trial.params["learning_rate"],
                "weight_decay": study.best_trial.params["weight_decay"],
            }

            self.fulltrain(best_cfg)

    def debug(self):
        print("Debug mode.")
        debug_config = {
            "channel_sizes": [16, 32, 64],
            "kernel_size": 3,
            "num_heads": 4,
            "dropout": 0.6,
            "horizon": 320,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4
        }

        debug_loader = BeatHarvest(trainCohort, SLIDING_WINDOW, debug_config["horizon"], debug_config["batch_size"]).push()

        debug_model = VTATTEND(
            input_dim=input_dim,
            channel_sizes=debug_config["channel_sizes"],
            kernel_size=debug_config["kernel_size"],
            num_heads=debug_config["num_heads"],
            dropout=debug_config["dropout"],
            horizon=debug_config["horizon"]
        )

        debug_trainer = Cardiobot(
            debug_model,
            debug_loader,
            debug_loader,
            lr=debug_config["learning_rate"],
            weight_decay=debug_config["weight_decay"]
        )

        debug_trainer.fit(num_epochs=10, pre_epochs=10, patience=0, ssl=True)
        return

# ---------------------------------------------------------------------------------/
def main():
    DEBUG = False
    pipeline = Pipeline()
    if not DEBUG:
        pipeline.controlpanel()
    else:
        pipeline.debug()

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