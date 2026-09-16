import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.cross_correlation import correlate, xcorr_max
import matplotlib.dates as mdates
from obspy.signal.trigger import classic_sta_lta, trigger_onset, recursive_sta_lta
import matplotlib.pyplot as plt


client = Client("https://seiscomp.alertnepal.online")


start = UTCDateTime("2026-08-26T02:52:00")
end   = UTCDateTime("2026-08-26T02:53:00")


st_kkn = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)

st_evn = client.get_waveforms(network="IO", station="EVN", location="*", channel="BHZ", starttime=start, endtime=end)

tr_kkn = st_kkn[0]
tr_evn = st_evn[0]

tr_kkn.detrend("demean").detrend("linear")
tr_evn.detrend("demean").detrend("linear")

tr_kkn.filter("bandpass", freqmin=2.0, freqmax=8.0)
tr_evn.filter("bandpass", freqmin=2.0, freqmax=8.0)

fs_kkn = tr_kkn.stats.sampling_rate
fs_evn = tr_evn.stats.sampling_rate

cft_kkn = recursive_sta_lta(tr_kkn.data, int(0.5 * fs_kkn), int(5.0 * fs_kkn))
cft_evn = recursive_sta_lta(tr_evn.data, int(0.5 * fs_evn), int(5.0 * fs_evn))
# sta_seconds = 1
# lta_seconds = 10

# # KKN
# sta_kkn = int(sta_seconds * fs_kkn)
# lta_kkn = int(lta_seconds * fs_kkn)

# cft_kkn = classic_sta_lta(
#     tr_kkn.data,
#     sta_kkn,
#     lta_kkn
# )

# # EVN
# sta_evn = int(sta_seconds * fs_evn)
# lta_evn = int(lta_seconds * fs_evn)

# cft_evn = classic_sta_lta(
#     tr_evn.data,
#     sta_evn,
#     lta_evn
# )

# Triggers
triggers_kkn = trigger_onset(cft_kkn, 3, 1)
triggers_evn = trigger_onset(cft_evn, 3, 1.0)

if len(triggers_kkn) > 0:
    p_pick_idx = triggers_kkn[0][0]  # First trigger index
    p_pick_sec = p_pick_idx / fs_kkn
    print(f"KKN P-wave onset: {p_pick_sec:.2f} s")

if len(triggers_evn) > 0:
    p_pick_idx = triggers_evn[0][0]  # First trigger index
    p_pick_sec = p_pick_idx / fs_evn
    print(f"EVN P-wave onset: {p_pick_sec:.2f} s")


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(tr_kkn.times(), tr_kkn.data)
axes[0].set_title("KKN BHZ")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

axes[1].plot(tr_evn.times(), tr_evn.data)
axes[1].set_title("EVN BHZ")
axes[1].set_xlabel("Seconds from 02:52:00 UTC")
axes[1].set_ylabel("Amplitude")
axes[1].grid()

plt.tight_layout()
plt.show()