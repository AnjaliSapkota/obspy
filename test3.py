import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows, coherence as _coherence, csd as _csd, correlate, correlation_lags

from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

SERVER = "ring.wscada.net:18000"

WINDOW = 20.0          # Seconds to compare
MAX_DELAY = 15.0        # Search lag range (-15.0 to +15.0 seconds)
BUFFER_LEN = 60.0      # Maximum seconds of data to retain in memory

# Preprocessing frequency range (Hz)
FREQ_MIN = 1.0
FREQ_MAX = 10.0

# CORR_TRIGGER_THRESHOLD = 0.5
# COHERENCE_THRESHOLD = 0.5
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
    n = min(len(s1), len(s2))
    if n == 0:
        return np.nan, np.nan

    s1, s2 = s1[:n], s2[:n]

    s1 = (s1 - np.mean(s1)) * windows.tukey(n, alpha=0.02)
    s2 = (s2 - np.mean(s2)) * windows.tukey(n, alpha=0.02)

    std1, std2 = np.std(s1), np.std(s2)
    if std1 == 0 or std2 == 0:
        return np.nan, np.nan

    # correlate(s2, s1): Positive lag means s2 is delayed relative to s1
    corr_full = correlate(s2, s1, mode='full') / (n * std1 * std2)
    lags_full = correlation_lags(n, n, mode='full')

    max_lag_samples = int(np.round(max_delay * fs))
    valid_mask = (lags_full >= -max_lag_samples) & (lags_full <= max_lag_samples)

    lags = lags_full[valid_mask]
    corr = corr_full[valid_mask]

    if len(corr) == 0:
        return np.nan, np.nan

    best_idx = np.argmax(np.abs(corr))
    best_corr = corr[best_idx]
    best_lag_seconds = lags[best_idx] / fs

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
    
    # Negate angle so positive phase lag matches positive time lag (s2 arriving after s1)
    phase = -np.angle(Pxy[nonzero])
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

    return CoherenceResult(
        freqs=freqs, coherence=Cxy, phase_lag_seconds=phase_lag,
        band_freqs=band_freqs, band_coherence=band_coh, band_phase_lag_seconds=band_lag,
        mean_coherence_in_band=mean_coh, lag_std_in_band=lag_std,
        n_coherent_bins=n_reliable, passed=True, reason="computed",
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
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))

        self.line_wave_11, = self.ax1.plot([], [], label="EQM11")
        self.line_wave_24, = self.ax1.plot([], [], label="EQM24")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Normalized Amplitude")
        self.ax1.legend(loc="upper right")
        self.ax1.grid(True)

        self.line_psd_11, = self.ax2.semilogy([], [], label="EQM11")
        self.line_psd_24, = self.ax2.semilogy([], [], label="EQM24")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Power")
        self.ax2.set_xlim(0, 20)
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
                # coherence_threshold=COHERENCE_THRESHOLD,
                # max_lag_std_seconds=MAX_LAG_STD_SECONDS,
            )
        except Exception as e:
            pass

        if diag is not None and len(diag.band_phase_lag_seconds) > 0:
            mean_phase_lag_s = np.nanmean(diag.band_phase_lag_seconds)
            phase_rads = 2 * np.pi * diag.band_freqs * diag.band_phase_lag_seconds
            mean_phase_lag_rad = np.nanmean(phase_rads)
            phase_lag_str = f"{mean_phase_lag_s:+.3f} s ({mean_phase_lag_rad / np.pi:+.2f}π rad)"
        else:
            phase_lag_str = "n/a"

        now_str = datetime.now().strftime("%H:%M:%S")
        utc_str = window_end.strftime("%H:%M:%S.%f")[:-3]

        print(
            f"[{now_str} | {utc_str} UTC] | "
            f"Window: {actual_window:.2f} s | "
            f"Fs: {fs:.1f} Hz | "
            f"Lag: {lag:+.3f} s | "
            f"Corr: {corr:.3f} | "
            f"Phase Lag: {phase_lag_str}"
        )

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
            f"EQM11 vs EQM24 | Time Lag = {lag:+.3f} s | Phase Lag = {phase_lag_str} | Corr = {corr:.3f}"
        )

        f11, p11 = welch(x11_norm, fs=fs, nperseg=min(1024, n))
        f24, p24 = welch(x24_norm, fs=fs, nperseg=min(1024, n))

        self.line_psd_11.set_data(f11, p11)
        self.line_psd_24.set_data(f24, p24)
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