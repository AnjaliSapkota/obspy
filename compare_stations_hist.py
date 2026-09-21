import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.dates as mdates
from scipy.signal import windows

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:52:00")
end   = UTCDateTime("2026-08-26T02:54:00")


st_a = client.get_waveforms(network="NK",station="KKN",location="*",channel="BHZ",starttime=start,endtime=end)

st_b = client.get_waveforms(
    network="IO",
    station="EVN",
    location="*",
    channel="BHZ",
    starttime=start,
    endtime=end
)


tr_a = st_a[0]
tr_b = st_b[0]


fs_a = tr_a.stats.sampling_rate
fs_b = tr_b.stats.sampling_rate

print("Original sampling rates:")
print("KKN:", fs_a, "Hz")
print("EVN:", fs_b, "Hz")


if fs_a != fs_b:
    tr_a.resample(fs_b)


# shift_seconds = 13.74
# tr_b.stats.starttime -= shift_seconds

common_start = max(tr_a.stats.starttime,tr_b.stats.starttime)

common_end = min(
    tr_a.stats.endtime,
    tr_b.stats.endtime
)

tr_a = tr_a.slice(common_start,common_end)

tr_b = tr_b.slice(common_start,common_end)


tr_a.filter("bandpass",freqmin=0.5,freqmax=8.0)

tr_b.filter("bandpass",freqmin=0.5,freqmax=8.0)


fs = tr_a.stats.sampling_rate

print("\nFinal sampling rate:", fs, "Hz")

print("Comparison duration:",tr_a.stats.endtime - tr_a.stats.starttime,"seconds")


n = min(len(tr_a.data),len(tr_b.data))

a = tr_a.data[:n].astype(float)
b = tr_b.data[:n].astype(float)

a -= np.mean(a)
b -= np.mean(b)


taper = windows.tukey(n,alpha=0.02)

a *= taper
b *= taper

max_shift_seconds = 60.0

max_shift_samples = int(round(max_shift_seconds * fs))

print("\nMaximum correlation shift:",max_shift_seconds,"seconds")


# Full linear correlation requires: 2*n-1 samples.

n_corr = 2 * n - 1

n_fft = 1 << (n_corr - 1).bit_length()


# Zero pad
A = np.zeros(n_fft)
B = np.zeros(n_fft)

A[:n] = a
B[:n] = b


# FFT
FA = np.fft.rfft(A)
FB = np.fft.rfft(B)


# Cross-spectrum
cross_spectrum = FA * np.conj(FB)


# IFFT
corr_circular = np.fft.irfft(
    cross_spectrum,
    n=n_fft
)


corr_full = np.concatenate(
    (
        corr_circular[-(n - 1):],
        corr_circular[:n]
    )
)

# Corresponding lags:
#
# -(n-1) ... -1  0  +1 ... +(n-1)

lags_full = np.arange(
    -(n - 1),
    n
)


mask = (
    (lags_full >= -max_shift_samples)
    &
    (lags_full <= max_shift_samples)
)

lags_samples = lags_full[mask]

corr = corr_full[mask]

normalized_cc = np.zeros_like(
    corr,
    dtype=float
)

for i, lag in enumerate(lags_samples):

    if lag >= 0:

        # A[lag:] compared with B[:n-lag]

        aa = a[lag:]

        bb = b[:n-lag]

    else:

        shift = -lag

        # A[:n-shift] compared with B[shift:]

        aa = a[:n-shift]

        bb = b[shift:]


    denominator = np.sqrt(
        np.sum(aa ** 2)
        *
        np.sum(bb ** 2)
    )


    if denominator > 0:

        normalized_cc[i] = (
            corr[i] / denominator
        )


peak_index = np.argmax(np.abs(normalized_cc))

best_lag_samples = (
    lags_samples[peak_index]
)

best_lag_seconds = (
    best_lag_samples / fs
)

max_corr = (
    normalized_cc[peak_index]
)

print(
    "Lag in samples:",
    best_lag_samples
)

print(
    "Lag in seconds:",
    best_lag_seconds
)

print(
    "Correlation coefficient:",
    max_corr
)

time_a = tr_a.times("matplotlib")
time_b = tr_b.times("matplotlib")


plt.figure(figsize=(14, 7))


plt.subplot(2, 1, 1)

plt.plot(
    time_a,
    tr_a.data
)

plt.ylabel("Amplitude")

plt.title(
    "Station A: NK.KKN.*.BHZ"
)

plt.grid(True)

plt.gca().xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M:%S")
)


plt.subplot(2, 1, 2)

plt.plot(
    time_b,
    tr_b.data
)

plt.xlabel("Time")

plt.ylabel("Amplitude")

plt.title(
    "Station B: IO.EVN.*.BHZ"
)

plt.grid(True)

plt.gca().xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M:%S")
)


plt.tight_layout()

plt.show()


lags_seconds = (
    lags_samples / fs
)


plt.figure(figsize=(12, 5))

plt.plot(
    lags_seconds,
    normalized_cc,
    label="Normalized Cross-Correlation"
)


plt.axvline(
    best_lag_seconds,
    linestyle="--",
    label=(
        f"Maximum = "
        f"{best_lag_seconds:.3f} s "
        f"(CC = {max_corr:.3f})"
    )
)


plt.axhline(
    0,
    linestyle=":"
)


plt.xlabel(
    "Lag / Time Delay (seconds)"
)

plt.ylabel(
    "Normalized Correlation Coefficient"
)

plt.title(
    "FFT/IFFT Waveform Cross-Correlation "
    "(KKN vs EVN)"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()