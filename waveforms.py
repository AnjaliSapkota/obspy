import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import spectrogram

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:45:00")
end   = UTCDateTime("2026-08-26T03:00:00")
event_time = UTCDateTime("2026-08-26T02:52:00")


kkn = client.get_waveforms("NK", "KKN", "*", "BHZ", start, end)
evn = client.get_waveforms("IO", "EVN", "*", "BHZ", start, end)

kkn.merge(method=1, fill_value="interpolate")
evn.merge(method=1, fill_value="interpolate")

# kkn.detrend("demean").detrend("linear")
# evn.detrend("demean").detrend("linear")

kkn_trace = kkn[0]
evn_trace = evn[0]

kkn_data = kkn_trace.data.astype(float)
evn_data = evn_trace.data.astype(float)

kkn_fs = kkn_trace.stats.sampling_rate
evn_fs = evn_trace.stats.sampling_rate

# Spectrograms
kkn_nperseg = int(20 * kkn_fs)
evn_nperseg = int(20 * evn_fs)

kkn_freq, kkn_time, kkn_Sxx = spectrogram(
    kkn_data,
    fs=kkn_fs,
    nperseg=kkn_nperseg,
    noverlap=kkn_nperseg // 2
)

evn_freq, evn_time, evn_Sxx = spectrogram(
    evn_data,
    fs=evn_fs,
    nperseg=evn_nperseg,
    noverlap=evn_nperseg // 2
)

kkn_Sxx_db = 10 * np.log10(kkn_Sxx + 1e-20)
evn_Sxx_db = 10 * np.log10(evn_Sxx + 1e-20)

start_matdate = start.matplotlib_date
kkn_spec_times = start_matdate + (kkn_time / 86400.0)
evn_spec_times = start_matdate + (evn_time / 86400.0)


fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# KKN WAVEFORM
axes[0].plot(
    kkn_trace.times("matplotlib"),
    kkn_data,
    linewidth=0.6,
    color="tab:orange"
)
axes[0].set_title("NK.KKN.BHZ — Waveform")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

# EVN WAVEFORM
axes[1].plot(
    evn_trace.times("matplotlib"),
    evn_data,
    linewidth=0.6,
    color="tab:blue"
)
axes[1].set_title("IO.EVN.BHZ — Waveform")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)

# KKN SPECTROGRAM
pcm1 = axes[2].pcolormesh(
    kkn_spec_times,
    kkn_freq,
    kkn_Sxx_db,
    shading="gouraud",
    cmap="viridis"
)
axes[2].set_ylim(0, 10)
axes[2].set_title("NK.KKN.BHZ — Spectrogram")
axes[2].set_ylabel("Frequency (Hz)")

# EVN SPECTROGRAM
pcm2 = axes[3].pcolormesh(
    evn_spec_times,
    evn_freq,
    evn_Sxx_db,
    shading="gouraud",
    cmap="viridis"
)
axes[3].set_ylim(0, 10)
axes[3].set_title("IO.EVN.BHZ — Spectrogram")
axes[3].set_ylabel("Frequency (Hz)")
axes[3].set_xlabel("UTC Time")

for ax in axes:
    ax.axvline(
        event_time.matplotlib_date,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8
    )

divider2 = make_axes_locatable(axes[2])
cax1 = divider2.append_axes("right", size="1.5%", pad=0.1)
fig.colorbar(pcm1, cax=cax1, label="Power (dB)")

divider3 = make_axes_locatable(axes[3])
cax2 = divider3.append_axes("right", size="1.5%", pad=0.1)
fig.colorbar(pcm2, cax=cax2, label="Power (dB)")

# Hide dummy colorbar spaces for waveform subplots to ensure exact alignment
for ax_idx in [0, 1]:
    divider = make_axes_locatable(axes[ax_idx])
    cax_dummy = divider.append_axes("right", size="1.5%", pad=0.1)
    cax_dummy.axis("off")

# Enforce uniform horizontal time bounds
axes[0].set_xlim(start.matplotlib_date, end.matplotlib_date)

# Format Time Axis
axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
fig.autofmt_xdate()

fig.suptitle(
    "KKN and EVN Waveforms + Spectrograms Around 02:52 UTC",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()