import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import hilbert, windows, correlate, correlation_lags
from obspy.geodetics import gps2dist_azimuth

# 1. Fetch Waveforms
client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:30:00")
end   = UTCDateTime("2026-08-26T03:00:00")

target_fs = 20.0
freqmin = 1.0
freqmax = 8.0
max_shift_seconds = 30.0

st_kkn = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)
st_evn = client.get_waveforms(network="IO", station="EVN", location="*", channel="BHZ", starttime=start, endtime=end)
st_eqm10 = client.get_waveforms(network="NP", station="EQM10", location="*", channel="HNZ", starttime=start, endtime=end)

# Merge gaps or segments
st_kkn.merge(method=1, fill_value="interpolate")
st_evn.merge(method=1, fill_value="interpolate")
st_eqm10.merge(method=1, fill_value="interpolate")

tr_kkn = st_kkn[0]
tr_evn = st_evn[0]
tr_eqm10 = st_eqm10[0]

# Station metadata & coordinates
inv_kkn = client.get_stations(network="NK", station="KKN", level="station")
inv_evn = client.get_stations(network="IO", station="EVN", level="station")
inv_eqm10 = client.get_stations(network="NP", station="EQM10", level="station")

def extract_station_coords(inv):
    sta = inv[0][0]
    return {"latitude": sta.latitude, "longitude": sta.longitude}

coord_kkn = extract_station_coords(inv_kkn)
coord_evn = extract_station_coords(inv_evn)
coord_eqm10 = extract_station_coords(inv_eqm10)

# Fixed station_distance function
def station_distance(coord1, coord2):
    dist_m, az, baz = gps2dist_azimuth(
        coord1["latitude"],
        coord1["longitude"],
        coord2["latitude"],
        coord2["longitude"]  # Corrected bracket syntax
    )
    return dist_m / 1000.0

dist_kkn_evn = station_distance(coord_kkn, coord_evn)
dist_kkn_eqm10 = station_distance(coord_kkn, coord_eqm10)
dist_evn_eqm10 = station_distance(coord_evn, coord_eqm10)

print("--- STATION DISTANCES ---")
print(f"KKN - EVN   : {dist_kkn_evn:.2f} km")
print(f"KKN - EQM10 : {dist_kkn_eqm10:.2f} km")
print(f"EVN - EQM10 : {dist_evn_eqm10:.2f} km\n")

# Preprocessing
def preprocess(tr):
    tr_proc = tr.copy()
    tr_proc.detrend("linear")
    tr_proc.detrend("demean")
    tr_proc.taper(max_percentage=0.05, type="cosine")
    tr_proc.interpolate(sampling_rate=target_fs, method="weighted_average_slopes")
    tr_proc.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
    return tr_proc

tr_kkn = preprocess(tr_kkn)
tr_evn = preprocess(tr_evn)
tr_eqm10 = preprocess(tr_eqm10)

# Align to common time window
common_start = max(tr_kkn.stats.starttime, tr_evn.stats.starttime, tr_eqm10.stats.starttime)
common_end = min(tr_kkn.stats.endtime, tr_evn.stats.endtime, tr_eqm10.stats.endtime)

tr_kkn.trim(common_start, common_end)
tr_evn.trim(common_start, common_end)
tr_eqm10.trim(common_start, common_end)

# Delay Calculation Function
def calculate_delay(tr1, tr2, name1, name2):
    raw1 = tr1.data.astype(float)
    raw2 = tr2.data.astype(float)

    min_len = min(len(raw1), len(raw2))
    raw1 = raw1[:min_len]
    raw2 = raw2[:min_len]

    fs = tr1.stats.sampling_rate

    # Envelopes via Hilbert Transform
    env1 = np.abs(hilbert(raw1))
    env2 = np.abs(hilbert(raw2))

    crop = int(0.05 * len(env1))
    env1 = env1[crop:-crop] - np.mean(env1[crop:-crop])
    env2 = env2[crop:-crop] - np.mean(env2[crop:-crop])

    taper = windows.tukey(len(env1), alpha=0.05)
    a = env1 * taper
    b = env2 * taper

    corr_full = correlate(a, b, mode="full")
    lags_full = correlation_lags(len(a), len(b), mode="full")
    norm_factor = np.sqrt(np.sum(a**2) * np.sum(b**2))
    normalized_cc_full = corr_full / norm_factor

    max_shift_samples = int(round(max_shift_seconds * fs))
    mask = (lags_full >= -max_shift_samples) & (lags_full <= max_shift_samples)

    lags_samples = lags_full[mask]
    normalized_cc = normalized_cc_full[mask]
    lags_seconds = lags_samples / fs

    peak_index = np.argmax(normalized_cc)
    best_lag_seconds = lags_seconds[peak_index]
    max_corr = normalized_cc[peak_index]

    return {
        "name1": name1,
        "name2": name2,
        "lag": best_lag_seconds,
        "correlation": max_corr,
        "lags": lags_seconds,
        "cc": normalized_cc
    }

# Compute delays
result_kkn_evn = calculate_delay(tr_kkn, tr_evn, "KKN", "EVN")
result_kkn_eqm10 = calculate_delay(tr_kkn, tr_eqm10, "KKN", "EQM10")
result_evn_eqm10 = calculate_delay(tr_evn, tr_eqm10, "EVN", "EQM10")

lag_kkn_evn = result_kkn_evn["lag"]
lag_kkn_eqm10 = result_kkn_eqm10["lag"]
lag_evn_eqm10 = result_evn_eqm10["lag"]

print("--- THREE-STATION TIME-DELAY RESULTS ---")
print(f"KKN → EVN     : {lag_kkn_evn:+.3f} s")
print(f"KKN → EQM10   : {lag_kkn_eqm10:+.3f} s")
print(f"EVN → EQM10   : {lag_evn_eqm10:+.3f} s")

predicted_kkn_eqm10 = lag_kkn_evn + lag_evn_eqm10
consistency_error = lag_kkn_eqm10 - predicted_kkn_eqm10

print("\n--- CONSISTENCY CHECK ---")
print(f"KKN→EVN + EVN→EQM10 = {predicted_kkn_eqm10:+.3f} s")
print(f"Measured KKN→EQM10   = {lag_kkn_eqm10:+.3f} s")
print(f"Consistency error     = {consistency_error:+.3f} s")

# # Plotting
# fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
# results = [result_kkn_evn, result_kkn_eqm10, result_evn_eqm10]

# for ax, res in zip(axes, results):
#     ax.plot(res["lags"], res["cc"], color='tab:blue')
#     ax.axvline(res["lag"], color='tab:red', linestyle="--", 
#                label=f"Lag = {res['lag']:+.3f} s, CC = {res['correlation']:.3f}")
#     ax.axvline(0, color='black', linestyle=":")
#     ax.set_ylabel("Correlation")
#     ax.set_title(f"{res['name1']} vs {res['name2']}")
#     ax.legend(loc="upper right")
#     ax.grid(True, alpha=0.3)

# axes[-1].set_xlabel("Lag (seconds)")
# plt.tight_layout()
# plt.show()

# # 6. Spectral & Phase Analysis
# freq_a, psd_a = welch(raw_a, fs=fs, window="hann", nperseg=int(20 * fs), noverlap=int(10 * fs))
# freq_b, psd_b = welch(raw_b, fs=fs, window="hann", nperseg=int(20 * fs), noverlap=int(10 * fs))

# f_a, t_a, Sxx_a = spectrogram(raw_a, fs=fs, window="hann", nperseg=int(10 * fs), noverlap=int(8 * fs), scaling="density")
# f_b, t_b, Sxx_b = spectrogram(raw_b, fs=fs, window="hann", nperseg=int(10 * fs), noverlap=int(8 * fs), scaling="density")

# f_coh, Cxy = coherence(raw_a, raw_b, fs=fs, window="hann", nperseg=int(20 * fs), noverlap=int(10 * fs))

# # Cross-Spectral Density (CSD) & Phase Lag calculation
# f_csd, Pxy = csd(raw_a, raw_b, fs=fs, window="hann", nperseg=int(20 * fs), noverlap=int(10 * fs))

# phase_lag_rad = np.angle(Pxy)
# phase_lag_deg = np.degrees(phase_lag_rad)

# # Time delay per frequency: tau(f) = phase_rad / (2 * pi * f)
# time_delay_f = np.zeros_like(phase_lag_rad)
# time_delay_f[1:] = phase_lag_rad[1:] / (2 * np.pi * f_csd[1:])

# # Mask phase lag where coherence is low (< 0.3)
# masked_phase_deg = np.where(Cxy >= 0.3, phase_lag_deg, np.nan)

# # 7. Visualization
# fig, axes = plt.subplots(5, 1, figsize=(12, 16))

# # Time Domain Envelopes
# t = np.arange(len(env_a)) / fs
# axes[0].plot(t, env_a, label="KKN Envelope", color="tab:blue")
# axes[0].plot(t, env_b, label="EVN Envelope", color="tab:orange")
# axes[0].set_ylabel("Amplitude")
# axes[0].set_title("Time Domain — Hilbert Envelopes")
# axes[0].legend()
# axes[0].grid(True)

# # Cross-Correlation Function
# axes[1].plot(lags_seconds, normalized_cc, color="black")
# axes[1].axvline(best_lag_seconds, color="red", linestyle="--", label=f"Peak Lag: {best_lag_seconds:.3f} s (CC: {max_corr:.2f})")
# axes[1].set_xlabel("Time Lag (s)")
# axes[1].set_ylabel("Normalized CC")
# axes[1].set_title("Envelope Cross-Correlation")
# axes[1].legend()
# axes[1].grid(True)

# # Power Spectral Density
# axes[2].semilogy(freq_a, psd_a, label="KKN")
# axes[2].semilogy(freq_b, psd_b, label="EVN")
# axes[2].set_xlim(0, 10)
# axes[2].set_xlabel("Frequency (Hz)")
# axes[2].set_ylabel("PSD")
# axes[2].set_title("Frequency Domain — Power Spectral Density")
# axes[2].legend()
# axes[2].grid(True)

# # Spectrograms
# mesh1 = axes[3].pcolormesh(t_a, f_a, 10 * np.log10(Sxx_a + 1e-20), shading="auto", cmap="viridis")
# axes[3].set_ylim(0, 10)
# axes[3].set_ylabel("Frequency (Hz)")
# axes[3].set_title("KKN Spectrogram (dB)")
# fig.colorbar(mesh1, ax=axes[3], label="dB")

# mesh2 = axes[4].pcolormesh(t_b, f_b, 10 * np.log10(Sxx_b + 1e-20), shading="auto", cmap="viridis")
# axes[4].set_ylim(0, 10)
# axes[4].set_xlabel("Time (s)")
# axes[4].set_ylabel("Frequency (Hz)")
# axes[4].set_title("EVN Spectrogram (dB)")
# fig.colorbar(mesh2, ax=axes[4], label="dB")

# plt.tight_layout()
# plt.show()

# # Coherence & Phase Lag Plot
# fig, (ax_coh, ax_phase) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# # Coherence Subplot
# ax_coh.plot(f_coh, Cxy, color="purple")
# ax_coh.set_ylabel("Coherence")
# ax_coh.set_title("Frequency-Domain Coherence & Phase Lag — KKN vs EVN")
# ax_coh.set_ylim(0, 1)
# ax_coh.set_xlim(0, 10)
# ax_coh.grid(True)

# # Phase Lag Subplot
# ax_phase.plot(f_csd, phase_lag_deg, color="gray", alpha=0.4, linestyle="--", label="Raw Phase")
# ax_phase.plot(f_csd, masked_phase_deg, color="crimson", linewidth=2, label="Phase (Coherence ≥ 0.3)")
# ax_phase.axhline(0, color="black", linestyle=":", alpha=0.7)
# ax_phase.set_xlabel("Frequency (Hz)")
# ax_phase.set_ylabel("Phase Lag (Degrees)")
# ax_phase.set_ylim(-180, 180)
# ax_phase.set_xlim(0, 10)
# ax_phase.legend()
# ax_phase.grid(True)

# plt.tight_layout()
# plt.show()