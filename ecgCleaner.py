import pywt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis, kstest

class ECGProcessor:
    def __init__(self, lead_names, var_thresh=1e-6, kurt_thresh=(0.5, 10), n_components=3, fs=128):
        self.lead_names = lead_names
        self.var_thresh = var_thresh
        self.kurt_thresh = kurt_thresh
        self.n_components = n_components
        self.raw = None
        self.cleaned = None
        self.bad_leads = set()
        self.fs = fs

    def wizard(self, signal_matrix):
        self.raw = signal_matrix
        cleaned = signal_matrix.copy()
        bad_channels = []

        # Step 1: Check Einthoven consistency
        error, failed, limb_leads = self.validate_einthoven_leads(cleaned, self.lead_names)
        if failed:
            # print(f"Einthoven inconsistency detected! MAE = {error:.4f}")
            outlier = self.identify_einthoven_outlier(cleaned, self.lead_names)
            if outlier:
                cleaned = self.reconstruct_einthoven_lead(cleaned, self.lead_names, outlier)
                # print(f"Reconstructed {outlier} from other limb leads.")

        # Step 2: Repair augmented leads
        for lead in ["avr (mV)", "avl (mV)", "avf (mV)"]:
            if lead not in self.lead_names:
                continue
            idx = self.lead_names.index(lead)
            var_ok = np.var(cleaned[:, idx]) >= self.var_thresh
            kurt_ok = self.kurt_thresh[0] <= kurtosis(cleaned[:, idx]) <= self.kurt_thresh[1]
            if not (var_ok and kurt_ok):
                cleaned = self.reconstruct_augmented_lead(cleaned, self.lead_names, lead)
                # print(f"Reconstructed {lead} (preemptive)")

        # Step 3: Detect statistical outliers
        variances = np.var(cleaned, axis=0)
        #kurts = kurtosis(cleaned, axis=0)
        kurts = kurtosis(cleaned, axis=0, fisher=False)

        for i, (v, k) in enumerate(zip(variances, kurts)):
            # print(f"Ch {i} | var={v:.3e}, kurt={k:.1f} ", end='')
            if v < self.var_thresh:
                # print("-> low variance")
                bad_channels.append(i)
            elif not (self.kurt_thresh[0] <= k <= self.kurt_thresh[1]):
                # print("-> abnormal kurtosis")
                bad_channels.append(i)
            else:
                True

        self.cleaned = cleaned
        self.bad_leads = set(bad_channels)
        cleaned = self.wavelets(cleaned)
        self.cleaned = cleaned

        return cleaned, self.bad_leads
    
    def wavelets(self, values=None):   
        # Wavelet-based denoising for each lead
        denoised = np.zeros_like(values)
        wavelet = 'db4'
        for i in range(values.shape[1]):
            signal = values[:, i]

            # Determine maximum decomposition level
            max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
            coeffs = pywt.wavedec(signal, wavelet, level=max_level)

            # Estimate noise sigma from the finest detail coefficients
            detail_coeff = coeffs[-1]
            sigma = np.median(np.abs(detail_coeff)) / 0.6745

            # Universal threshold
            uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

            # Threshold all detail coefficients
            thresholded = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]

            # Reconstruct the denoised signal
            reconstructed = pywt.waverec(thresholded, wavelet)

            # Truncate to original length in case of padding
            denoised[:, i] = reconstructed[:len(signal)]

        return denoised

    def validate_einthoven_leads(self, signal_matrix, lead_names, threshold=0.15):
        try:
            idx_I = lead_names.index("i (mV)")
            idx_II = lead_names.index("ii (mV)")
            idx_III = lead_names.index("iii (mV)")
        except ValueError:
            return None, False, []

        I, II, III = signal_matrix[:, idx_I], signal_matrix[:, idx_II], signal_matrix[:, idx_III]
        error = np.mean(np.abs((I + III) - II))
        failed = error > threshold
        return error, failed, [idx_I, idx_II, idx_III]

    def identify_einthoven_outlier(self, signal_matrix, lead_names):
        try:
            idx_I = lead_names.index("i (mV)")
            idx_II = lead_names.index("ii (mV)")
            idx_III = lead_names.index("iii (mV)")
        except ValueError:
            return None

        I = signal_matrix[:, idx_I]
        II = signal_matrix[:, idx_II]
        III = signal_matrix[:, idx_III]

        I_est = II - III
        II_est = I + III
        III_est = II - I

        err_I = np.mean(np.abs(I - I_est))
        err_II = np.mean(np.abs(II - II_est))
        err_III = np.mean(np.abs(III - III_est))

        errs = np.array([err_I, err_II, err_III])
        mean_error = np.mean(errs)

        if mean_error < 0.3 or np.allclose(errs, errs[0], atol=1e-5):
            return None

        errors = {"i (mV)": err_I, "ii (mV)": err_II, "iii (mV)": err_III}
        likely_bad = max(errors, key=errors.get)

        return likely_bad

    def reconstruct_einthoven_lead(self, signal_matrix, lead_names, bad_lead, bad_leads=None):
        if bad_leads is None:
            bad_leads = self.bad_leads

        try:
            idx_I = lead_names.index("i (mV)")
            idx_II = lead_names.index("ii (mV)")
            idx_III = lead_names.index("iii (mV)")
        except ValueError:
            return signal_matrix

        if bad_lead == "i (mV)" and all(idx not in bad_leads for idx in [idx_II, idx_III]):
            signal_matrix[:, idx_I] = signal_matrix[:, idx_II] - signal_matrix[:, idx_III]
        elif bad_lead == "ii (mV)" and all(idx not in bad_leads for idx in [idx_I, idx_III]):
            signal_matrix[:, idx_II] = signal_matrix[:, idx_I] + signal_matrix[:, idx_III]
        elif bad_lead == "iii (mV)" and all(idx not in bad_leads for idx in [idx_I, idx_II]):
            signal_matrix[:, idx_III] = signal_matrix[:, idx_II] - signal_matrix[:, idx_I]
        return signal_matrix

    def reconstruct_augmented_lead(self, signal_matrix, lead_names, bad_lead, bad_leads=None):
        if bad_leads is None:
            bad_leads = self.bad_leads

        try:
            idx_I = lead_names.index("i (mV)")
            idx_II = lead_names.index("ii (mV)")
            idx = lead_names.index(bad_lead)
        except ValueError:
            return signal_matrix

        if bad_lead == "avr (mV)":
            signal_matrix[:, idx] = -0.5 * (signal_matrix[:, idx_I] + signal_matrix[:, idx_II])
        elif bad_lead == "avl (mV)":
            signal_matrix[:, idx] = signal_matrix[:, idx_I] - 0.5 * signal_matrix[:, idx_II]
        elif bad_lead == "avf (mV)":
            signal_matrix[:, idx] = signal_matrix[:, idx_II] - 0.5 * signal_matrix[:, idx_I]
        return signal_matrix

    def interpolate_bad_channels(self, signal_matrix, bad_channels, method="pca", replace_good=False):
        cleaned = signal_matrix.copy()
        if not bad_channels:
            return cleaned

        lead_map = {name: i for i, name in enumerate(self.lead_names)}
        chest = ["v1 (mV)", "v2 (mV)", "v3 (mV)", "v4 (mV)", "v5 (mV)", "v6 (mV)"]
        chest_idxs = [lead_map[l] for l in chest if l in lead_map]
        good_chest = [i for i in chest_idxs if i not in bad_channels]
        bad_chest = [i for i in chest_idxs if i in bad_channels]

        if method == "mean":
            for i in bad_chest:
                if good_chest:
                    cleaned[:, i] = np.mean(signal_matrix[:, good_chest], axis=1)
        elif method == "pca":
            cmr_signal = self.common_mode_reject_chest_leads(signal_matrix.copy(), self.lead_names, bad_channels)
            if len(good_chest) < 2:
                return self.interpolate_bad_channels(signal_matrix, bad_channels, method="mean")
            pca = PCA(n_components=min(len(good_chest), self.n_components))
            comp = pca.fit_transform(cmr_signal[:, good_chest])
            rec = pca.inverse_transform(comp)

            for i in bad_chest:
                model = LinearRegression()
                model.fit(rec, cmr_signal[:, i])
                cleaned[:, i] = model.predict(rec)

            if replace_good:
                for j, i in enumerate(good_chest):
                    cleaned[:, i] = rec[:, j]

            cmr = np.mean(signal_matrix[:, good_chest], axis=1, keepdims=True)
            for idx in chest_idxs:
                cleaned[:, idx] += cmr[:, 0]
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return cleaned

    @staticmethod
    def common_mode_reject_chest_leads(signal_matrix, lead_names, bad_channels):
        modified = signal_matrix.copy()
        v_leads = ["v1 (mV)", "v2 (mV)", "v3 (mV)", "v4 (mV)", "v5 (mV)", "v6 (mV)"]
        v_indices = [lead_names.index(lead) for lead in v_leads if lead in lead_names]
        good_v_indices = [idx for idx in v_indices if idx not in bad_channels]

        if len(good_v_indices) < 2:
            return signal_matrix

        cmr = np.mean(signal_matrix[:, good_v_indices], axis=1, keepdims=True)
        for idx in v_indices:
            modified[:, idx] -= cmr[:, 0]

        return modified

    def compress_signal(self, signal_matrix):
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(signal_matrix)
    
    def check_and_transform_for_normality(self, metric_array, name="metric", force_log=False):
        metric_array = np.asarray(metric_array)
        epsilon = 1e-12  # to avoid log(0)
        is_positive = np.all(metric_array > 0)

        if force_log and is_positive:
            log_metric = np.log(metric_array + epsilon)
            # print(f"Forcing log-transform for {name}")
            return log_metric, "log"

        # Normality test
        mean_, std_ = np.mean(metric_array), np.std(metric_array)
        stat, p = kstest(metric_array, 'norm', args=(mean_, std_))

        if p > 0.05:
            return metric_array, "none"

        if is_positive:
            log_metric = np.log(metric_array + epsilon)
            log_mean, log_std = np.mean(log_metric), np.std(log_metric)
            stat_log, p_log = kstest(log_metric, 'norm', args=(log_mean, log_std))

            if p_log > 0.05:
                return log_metric, "log"
            else:
                return metric_array, "none"
        else:
            return metric_array, "none"

    def shannon_entropy(self, signal):
        """Computes entropy of a signal."""
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = np.clip(hist, 1e-12, None)  # prevent log(0)
        return -np.sum(hist * np.log2(hist))

    def pipeline(self, signal_matrix, method="hybrid", threshold=2.5, window_size=5, action="interpolate", plot_summary=True):
        window_size = int(window_size * self.fs)
        n_samples, n_channels = signal_matrix.shape
        n_windows = n_samples // window_size
        noisy_windows = []

        powers, vars_, kurts_, ents = [], [], [], []

        for w in range(n_windows):
            start, end = w * window_size, (w + 1) * window_size
            window = signal_matrix[start:end]

            powers.append(np.mean(window ** 2))
            vars_.append(np.mean(np.var(window, axis=0)))
            kurts_.append(np.mean(kurtosis(window, axis=0, fisher=False)))
            ents.append(np.mean([self.shannon_entropy(window[:, ch]) for ch in range(n_channels)]))

        powers, _ = self.check_and_transform_for_normality(powers, "Power", force_log=True)
        vars_, _ = self.check_and_transform_for_normality(vars_, "Variance", force_log=True)
        kurts_, _ = self.check_and_transform_for_normality(kurts_, "Kurtosis")
        ents, _ = self.check_and_transform_for_normality(ents, "Entropy")

        def z_mask(arr):
            z = (arr - np.mean(arr)) / (np.std(arr) + 1e-10)
            return np.abs(z) > threshold

        mask_power = z_mask(powers)
        mask_var = z_mask(vars_)
        mask_kurt = z_mask(kurts_)
        mask_ent = z_mask(ents)

        # Union of masks (logical OR)
        noisy_mask = mask_power | mask_var | mask_kurt | mask_ent

        for i, is_noisy in enumerate(noisy_mask):
            if is_noisy:
                start = i * window_size
                end = start + window_size
                noisy_windows.append((start, end))

        ratio_noisy = np.sum(noisy_mask) * window_size / n_samples

        if plot_summary:
            metric_labels = ["Power", "Variance", "Kurtosis", "Entropy"]
            metric_data = [powers, vars_, kurts_, ents]
            plt.figure(figsize=(12, 8))
            for i, (data, label) in enumerate(zip(metric_data, metric_labels)):
                plt.subplot(2, 2, i+1)
                plt.hist(data, bins=30, alpha=0.7, edgecolor='k')
                plt.axvline(np.mean(data) + threshold*np.std(data), color='r', linestyle='--', label="Upper")
                plt.axvline(np.mean(data) - threshold*np.std(data), color='r', linestyle='--', label="Lower")
                plt.title(f"{label} Distribution")
                plt.xlabel(label)
                plt.ylabel("Count")
                plt.legend()
            plt.tight_layout()
            plt.show()

        # Getting the mask
        if action == "mask":
            window_mask = np.ones(n_samples, dtype=int)
            cleaned = signal_matrix.copy()
            for start, end in noisy_windows:
                #cleaned[start:end, :] = 0
                window_mask[start:end] = 0
        elif action == "interpolate":
            cleaned = self.interpolate_noisy_window_spline(signal_matrix, noisy_windows, noisy_mask)
        else:
            raise ValueError(f"Unknown action: {action}")
        
        return cleaned, noisy_windows, ratio_noisy, window_mask

    def interpolate_noisy_window_spline(self, signal_matrix, window_indices, noisy_mask):
            interpolated_signal = signal_matrix.copy()
            n_samples, n_channels = signal_matrix.shape

            for (start, end), is_noisy in zip(window_indices, noisy_mask):
                if not is_noisy:
                    continue
                for ch in range(n_channels):
                    # Identify valid boundary indices
                    left = start - 1 if start - 1 >= 0 else None
                    right = end if end < n_samples else None

                    if left is not None and right is not None:
                        idx_range = list(range(max(0, left - 3), left + 1)) + list(range(right, min(n_samples, right + 4)))
                        idx_range = sorted(set(idx_range))
                        if len(idx_range) < 4:
                            continue
                        x_known = np.array(idx_range)
                        y_known = signal_matrix[x_known, ch]

                        spline = CubicSpline(x_known, y_known)
                        x_interp = np.arange(start, end)
                        interpolated_signal[x_interp, ch] = spline(x_interp)

            return interpolated_signal