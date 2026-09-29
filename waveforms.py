import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import spectrogram

# Initialize FDSN client
client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:00:00")
end   = UTCDateTime("2026-08-26T04:00:00")
event_time = UTCDateTime("2026-08-26T02:52:00")

kkn = client.get_waveforms("NK", "KKN", "*", "BHZ", start, end)
evn = client.get_waveforms("IO", "EVN", "*", "BHZ", start, end)

# kkn_inv = client.get_stations(network="NK", station="KKN", location="*", channel="BHZ", level="response")
# evn_inv = client.get_stations(network="IO", station="EVN", location="*", channel="BHZ", level="response")

# pre_filt = (0.005, 0.006, 30.0, 35.0)
# kkn.remove_response(inventory=kkn_inv, output='DISP', pre_filt=pre_filt)
# evn.remove_response(inventory=evn_inv, output='DISP', pre_filt=pre_filt)

# for inv, name in [(kkn_inv, "KKN"), (evn_inv, "EVN")]:
#     print(f"\n{name}")
#     print(inv)

#     for network in inv:
#         for station in network:
#             for channel in station:
#                 print(
#                     f"{channel.code}: "
#                     f"response = {channel.response}"
#                 )

#                 if channel.response:
#                     print(
#                         "Number of response stages:",
#                         len(channel.response.response_stages)
#                     )

# Merge gaps & Preprocess before response removal
kkn.merge(method=1, fill_value="interpolate")
evn.merge(method=1, fill_value="interpolate")

kkn.detrend("demean").detrend("linear").taper(max_percentage=0.05, type="cosine")
evn.detrend("demean").detrend("linear").taper(max_percentage=0.05, type="cosine")

# # #  Remove Instrument Response (Converts raw ADC counts to Ground Velocity in m/s)
# # pre_filt = (0.005, 0.01, 10.0, 20.0)
# # # Try full stage response removal first; fall back to simple sensitivity division
# # try:
# #     kkn.remove_response(inventory=kkn_inv, output="VEL", pre_filt=pre_filt, water_level=60)
# # except IndexError:
# #     print("Full response stages missing for KKN, falling back to sensitivity division...")
# #     kkn.remove_sensitivity(inventory=kkn_inv)

# # try:
# #     evn.remove_response(inventory=evn_inv, output="VEL", pre_filt=pre_filt, water_level=60)
# # except IndexError:
# #     print("Full response stages missing for EVN, falling back to sensitivity division...")
# #     evn.remove_sensitivity(inventory=evn_inv)

kkn_trace = kkn[0]
evn_trace = evn[0]

kkn_data = kkn_trace.data.astype(float)
evn_data = evn_trace.data.astype(float)

kkn_fs = kkn_trace.stats.sampling_rate
evn_fs = evn_trace.stats.sampling_rate

# 5. Compute Spectrograms
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

# Convert PSD to dB relative to 1 (m/s)^2 / Hz
kkn_Sxx_db = 10 * np.log10(kkn_Sxx + 1e-20)
evn_Sxx_db = 10 * np.log10(evn_Sxx + 1e-20)

start_matdate = start.matplotlib_date
kkn_spec_times = start_matdate + (kkn_time / 86400.0)
evn_spec_times = start_matdate + (evn_time / 86400.0)

# 6. Plotting - 4 Vertical Subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

# --- 1. KKN WAVEFORM ---
axes[0].plot(
    kkn_trace.times("matplotlib"),
    kkn_data,
    linewidth=0.6,
    color="tab:orange"
)
axes[0].set_title("NK.KKN.BHZ — Waveform")
axes[0].set_ylabel("Velocity (m/s)")
axes[0].grid(True, alpha=0.3)

# --- 2. EVN WAVEFORM ---
axes[1].plot(
    evn_trace.times("matplotlib"),
    evn_data,
    linewidth=0.6,
    color="tab:blue"
)
axes[1].set_title("IO.EVN.BHZ — Waveform")
axes[1].set_ylabel("Velocity (m/s)")
axes[1].grid(True, alpha=0.3)

# --- 3. KKN SPECTROGRAM ---
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

# --- 4. EVN SPECTROGRAM ---
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

# --- Event Marker Line across all axes ---
for ax in axes:
    ax.axvline(
        event_time.matplotlib_date,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="02:52 UTC Event"
    )

# --- Align Subplot Widths using colorbar dummies ---
divider2 = make_axes_locatable(axes[2])
cax1 = divider2.append_axes("right", size="1.5%", pad=0.1)
fig.colorbar(pcm1, cax=cax1, label="Power [dB rel. (m/s)²/Hz]")

divider3 = make_axes_locatable(axes[3])
cax2 = divider3.append_axes("right", size="1.5%", pad=0.1)
fig.colorbar(pcm2, cax=cax2, label="Power [dB rel. (m/s)²/Hz]")

# Dummy colorbars for the top two axes to ensure identical plot area widths
for ax_idx in [0, 1]:
    divider = make_axes_locatable(axes[ax_idx])
    cax_dummy = divider.append_axes("right", size="1.5%", pad=0.1)
    cax_dummy.axis("off")

# Set bounds & formatting
axes[0].set_xlim(start.matplotlib_date, end.matplotlib_date)
axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
fig.autofmt_xdate()

fig.suptitle(
    "KKN and EVN Ground Motion (m/s) + Spectrograms Around 02:52 UTC",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

plt.tight_layout()
plt.show()