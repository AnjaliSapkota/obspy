import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import spectrogram

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-09-22T07:45:00")
end   = UTCDateTime("2026-09-22T07:55:00")

kkn = client.get_waveforms("NK", "KKN", "*", "BHZ", start, end)
evn = client.get_waveforms("IO", "EVN", "*", "BHZ", start, end)
eqm08 = client.get_waveforms("NP", "EQM08", "*", "EHZ", start, end)
eqm10 = client.get_waveforms("NP", "EQM10", "*", "EHZ", start, end)

# Store all stations together
streams = {"KKN": kkn,"EVN": evn,"EQM08": eqm08,"EQM10": eqm10}

# Preprocess all stations
for name, st in streams.items():
    st.merge(method=1, fill_value="interpolate")
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05,type="cosine")
    st.interpolate(sampling_rate=20, method="linear")
traces = {"KKN": kkn[0],"EVN": evn[0],"EQM08": eqm08[0],"EQM10": eqm10[0]}

# Get data and sampling rates
data = {}
sampling_rates = {}

for name, tr in traces.items():
    data[name] = tr.data.astype(float)
    sampling_rates[name] = tr.stats.sampling_rate
    print(f"{name}: "f"{tr.id}, "f"Fs={tr.stats.sampling_rate} Hz, "f"Samples={len(tr.data)}")

# waveform
fig_wave, axes_wave = plt.subplots(4,1,figsize=(15, 12),sharex=True)
station_order = ["KKN", "EVN", "EQM08", "EQM10"]
for ax, name in zip(axes_wave, station_order):
    tr = traces[name]
    ax.plot(tr.times("matplotlib"),data[name],linewidth=0.5)

    ax.set_title(f"{tr.id}")
    ax.set_ylabel("Amplitude")
    ax.grid(True,alpha=0.3)

axes_wave[-1].set_xlabel("UTC Time")

# Format time
axes_wave[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

axes_wave[0].set_xlim(start.matplotlib_date,end.matplotlib_date)

fig_wave.suptitle("Seismic Waveforms - 22 September 2026",fontsize=15,fontweight="bold")

plt.tight_layout()
plt.show()


# Spectogram

fig_spec, axes_spec = plt.subplots(4,1,figsize=(15, 14),
    sharex=True
)

for ax, name in zip(axes_spec, station_order):

    tr = traces[name]

    signal = data[name]
    fs = sampling_rates[name]

    # 20-second spectrogram window
    nperseg = int(20 * fs)

    # Make sure nperseg is not larger than signal
    nperseg = min(
        nperseg,
        len(signal)
    )

    freq, time, Sxx = spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2
    )

    # Convert PSD to dB
    Sxx_db = 10 * np.log10(
        Sxx + 1e-20
    )

    # Convert relative seconds to matplotlib dates
    spec_times = (
        start.matplotlib_date
        + time / 86400.0
    )

    pcm = ax.pcolormesh(
        spec_times,
        freq,
        Sxx_db,
        shading="gouraud",
        cmap="viridis"
    )

    # Display 0–10 Hz
    ax.set_ylim(
        0,
        10
    )

    ax.set_title(
        f"{tr.id}"
    )

    ax.set_ylabel(
        "Frequency (Hz)"
    )

    # Colorbar for each station
    cbar = fig_spec.colorbar(
        pcm,
        ax=ax,
        pad=0.01
    )

    cbar.set_label(
        "Power [dB]"
    )

axes_spec[-1].set_xlabel(
    "UTC Time"
)

axes_spec[-1].xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M:%S")
)

axes_spec[0].set_xlim(
    start.matplotlib_date,
    end.matplotlib_date
)

fig_spec.suptitle(
    "Seismic Spectrograms — 22 September 2026",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()