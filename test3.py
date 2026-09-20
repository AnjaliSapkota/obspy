import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows, coherence as _coherence, csd as _csd

from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

SERVER = "ring.wscada.net:18000"

WINDOW = 10.0          # Seconds to compare
MAX_DELAY = 2.0        # Search lag range (-2.0 to +2.0 seconds)
BUFFER_LEN = 60.0      # Maximum seconds of data to retain in memory

# Preprocessing frequency range (Hz)
FREQ_MIN = 1.0
FREQ_MAX = 10.0

CORR_TRIGGER_THRESHOLD = 0.5
COHERENCE_THRESHOLD = 0.5
MAX_LAG_STD_SECONDS = 0.05

stations = [
    {"net": "NP", "sta": "EQM11", "cha": "HNZ"},
    {"net": "NP", "sta": "EQM24", "cha": "HNZ"},
]


def cross_correlation(s1, s2, fs, max_delay, raise_on_invalid=False):
    if fs <= 0 or max_delay < 0:
        return np.nan, np.nan

    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    if len(s1) == 0 or len(s2) == 0:
        return np.nan, np.nan

    dt = 1.0 / fs
    n1, n2 = len(s1), len(s2)

    duration = min((n1 - 1) * dt, (n2 - 1) * dt)
    if max_delay > duration:
        max_delay = duration

    s1 = (s1 - np.mean(s1)) * windows.tukey(n1, alpha=0.02)
    s2 = (s2 - np.mean(s2)) * windows.tukey(n2, alpha=0.02)

    n_conv = n1 + n2 - 1
    n_fft = 1 << (n_conv - 1).bit_length()

    x1 = np.zeros(n_fft)
    x2 = np.zeros(n_fft)
    x1[:n1] = s1

    offset = n1 - 1
    x2[offset:offset + n2] = s2

    F1 = np.fft.fft(x1)
    F2 = np.fft.fft(x2)
    corr_full = np.fft.ifft(F1 * np.conj(F2)).real

    max_lag_samples = int(np.round(max_delay * fs))
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    indices = (lags - offset) % n_fft
    corr_raw = corr_full[indices]

    c1 = np.concatenate(([0.0], np.cumsum(s1 ** 2)))
    c2 = np.concatenate(([0.0], np.cumsum(s2 ** 2)))

    corr_normalized = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            start1, end1 = lag, min(n1, n2 + lag)
            start2, end2 = 0, end1 - start1
        else:
            start1, end1 = 0, min(n1, n2 + lag)
            start2 = -lag
            end2 = start2 + (end1 - start1)

        if end1 <= start1 or end2 <= start2:
            continue

        e1 = c1[end1] - c1[start1]
        e2 = c2[end2] - c2[start2]
        denom = np.sqrt(e1 * e2)
        if denom > 0:
            corr_normalized[i] = corr_raw[i] / denom

    if not np.any(corr_normalized):
        return np.nan, np.nan

    best_idx = np.argmax(np.abs(corr_normalized))
    best_corr = corr_normalized[best_idx]
    best_lag_seconds = lags[best_idx] / fs

    if abs(best_corr) > 1.0 + 1e-6:
        msg = f"Correlation coefficient out of bounds: {best_corr}."
        if raise_on_invalid:
            raise AssertionError(msg)
        warnings.warn(msg)
        return np.nan, np.nan

    return float(best_corr), float(best_lag_seconds)


@dataclass
class CoherenceResult:
    freqs: np.ndarray
    coherence: np.ndarray
    phase_lag_seconds: np.ndarray
    band_freqs: np.ndarray
    band_coherence: np.ndarray
    band_phase_lag_seconds: np.ndarray
    mean_coherence_in_band: float
    lag_std_in_band: float
    n_coherent_bins: int
    passed: bool
    reason: str


def spectral_coherence_check(
    s1, s2, fs, reference_lag_seconds,
    freq_band=(1.0, 10.0), coherence_threshold=0.5,
    max_lag_std_seconds=None, min_coherent_bins=3, nperseg=None,
):
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]

    if nperseg is None:
        nperseg = min(256, n)
    nperseg = max(8, min(nperseg, n))

    freqs, Cxy = _coherence(s1, s2, fs=fs, nperseg=nperseg)
    freqs_csd, Pxy = _csd(s1, s2, fs=fs, nperseg=nperseg)

    ref_lag = 0.0 if np.isnan(reference_lag_seconds) else reference_lag_seconds

    phase_lag = np.full_like(freqs, np.nan)
    nonzero = freqs > 0
    phase = np.angle(Pxy[nonzero])
    f_nz = freqs[nonzero]
    tau0 = phase / (2 * np.pi * f_nz)
    k = np.round((ref_lag - tau0) * f_nz)
    phase_lag[nonzero] = tau0 + k / f_nz

    fmin, fmax = freq_band
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_freqs = freqs[band_mask]
    band_coh = Cxy[band_mask]
    band_lag = phase_lag[band_mask]

    reliable = band_coh >= coherence_threshold
    n_reliable = int(np.sum(reliable))
    mean_coh = float(np.mean(band_coh)) if len(band_coh) else float("nan")

    if n_reliable >= min_coherent_bins:
        lag_std = float(np.std(band_lag[reliable]))
    else:
        lag_std = float("nan")

    passed = True
    reason = "computed"

    return CoherenceResult(
        freqs=freqs, coherence=Cxy, phase_lag_seconds=phase_lag,
        band_freqs=band_freqs, band_coherence=band_coh, band_phase_lag_seconds=band_lag,
        mean_coherence_in_band=mean_coh, lag_std_in_band=lag_std,
        n_coherent_bins=n_reliable, passed=passed, reason=reason,
    )


class MyClient(EasySeedLinkClient):

    def __init__(self, server):
        super().__init__(server, autoconnect=False)

        self.station_streams = {
            "EQM11": Stream(),
            "EQM24": Stream(),
        }

        self.last_plot_time = 0
        self.last_processed_utc = None

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 7))

        # Waveform subplot
        self.line_wave_11, = self.ax1.plot([], [], label="EQM11")
        self.line_wave_24, = self.ax1.plot([], [], label="EQM24")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Normalized Amplitude")
        self.ax1.legend(loc="upper right")
        self.ax1.grid(True)

        # Spectral phase lag subplot
        self.line_phase_lag, = self.ax2.plot([], [], 'o-', label="Per-Frequency Phase Lag (s)")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Phase Lag (s)")
        self.ax2.set_xlim(FREQ_MIN, FREQ_MAX)
        self.ax2.legend(loc="upper right")
        self.ax2.grid(True)

        plt.tight_layout()

    def on_data(self, trace):
        station = trace.stats.station

        if station not in self.station_streams:
            return

        self.station_streams[station] += trace

        try:
            self.station_streams[station].merge(method=1, fill_value="interpolate")
        except Exception:
            return

        st11 = self.station_streams["EQM11"].select(station="EQM11", channel="HNZ")
        st24 = self.station_streams["EQM24"].select(station="EQM24", channel="HNZ")

        if len(st11) == 0 or len(st24) == 0:
            return

        tr11 = st11[0]
        tr24 = st24[0]

        fs = tr11.stats.sampling_rate
        if abs(fs - tr24.stats.sampling_rate) > 0.001:
            return

        common_start = max(tr11.stats.starttime, tr24.stats.starttime)
        common_end = min(tr11.stats.endtime, tr24.stats.endtime)

        if common_end - common_start < WINDOW:
            return

        window_end = common_end
        window_start = window_end - WINDOW

        if self.last_processed_utc == window_end:
            return
        self.last_processed_utc = window_end

        trim_time = common_end - BUFFER_LEN
        self.station_streams["EQM11"].trim(starttime=trim_time)
        self.station_streams["EQM24"].trim(starttime=trim_time)

        try:
            a = tr11.slice(starttime=window_start, endtime=window_end).copy()
            b = tr24.slice(starttime=window_start, endtime=window_end).copy()

            a.filter("bandpass", freqmin=FREQ_MIN, freqmax=FREQ_MAX, corners=4, zerophase=True)
            b.filter("bandpass", freqmin=FREQ_MIN, freqmax=FREQ_MAX, corners=4, zerophase=True)
        except Exception:
            return

        x11 = a.data.astype(float)
        x24 = b.data.astype(float)

        n = min(len(x11), len(x24))
        if n < int(0.8 * WINDOW * fs):
            return

        x11, x24 = x11[:n], x24[:n]
        actual_window = n / fs

        x11_norm = (x11 - np.mean(x11)) / np.std(x11) if np.std(x11) > 0 else x11
        x24_norm = (x24 - np.mean(x24)) / np.std(x24) if np.std(x24) > 0 else x24

        try:
            corr, lag = cross_correlation(x11, x24, fs, MAX_DELAY)
        except Exception as e:
            corr, lag = np.nan, np.nan

        diag = None
        try:
            diag = spectral_coherence_check(
                x11, x24, fs,
                reference_lag_seconds=lag,
                freq_band=(FREQ_MIN, FREQ_MAX),
                coherence_threshold=COHERENCE_THRESHOLD,
                max_lag_std_seconds=MAX_LAG_STD_SECONDS,
            )
        except Exception as e:
            pass

        now_str = datetime.now().strftime("%H:%M:%S")
        utc_str = window_end.strftime("%H:%M:%S.%f")[:-3]

        if diag is not None and len(diag.band_phase_lag_seconds) > 0:
            mean_phase_lag = np.nanmean(diag.band_phase_lag_seconds)
            phase_lag_str = f"{mean_phase_lag:+.3f} s"
        else:
            phase_lag_str = "n/a"

        print("=" * 80)
        print(
            f"[{now_str} | {utc_str} UTC] | "
            f"Window: {actual_window:.2f}s | "
            f"XCorr Lag: {lag:+.3f}s | "
            f"Corr: {corr:.3f} | "
            f"Mean Phase Lag: {phase_lag_str}"
        )

        # Print per-frequency breakdown
        if diag is not None and len(diag.band_freqs) > 0:
            print("Per-Frequency Phase Lags:")
            for f, coh, p_lag in zip(diag.band_freqs, diag.band_coherence, diag.band_phase_lag_seconds):
                print(f"  -> {f:5.2f} Hz | Coherence: {coh:.2f} | Phase Lag: {p_lag:+.4f} s")
        print("=" * 80)

        current_time = datetime.now().timestamp()
        if current_time - self.last_plot_time < 0.5:
            return
        self.last_plot_time = current_time

        t = np.arange(n) / fs

        self.line_wave_11.set_data(t, x11_norm)
        self.line_wave_24.set_data(t, x24_norm)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax1.set_title(
            f"EQM11 vs EQM24 | XCorr Lag = {lag:+.3f} s | Mean Phase Lag = {phase_lag_str}"
        )

        if diag is not None and len(diag.band_freqs) > 0:
            self.line_phase_lag.set_data(diag.band_freqs, diag.band_phase_lag_seconds)
            self.ax2.relim()
            self.ax2.autoscale_view()

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


if __name__ == "__main__":
    client = MyClient(SERVER)
    client.conn.timeout = 10
    client.connect()

    for target in stations:
        client.select_stream(target["net"], target["sta"], target["cha"])

    try:
        client.run()
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    finally:
        client.close()