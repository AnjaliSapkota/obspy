import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.cross_correlation import correlate, xcorr_max
import matplotlib.dates as mdates
from obspy.signal.trigger import classic_sta_lta, trigger_onset


client = Client("https://seiscomp.alertnepal.online")


start = UTCDateTime("2026-08-26T02:52:00")
end   = UTCDateTime("2026-08-26T02:53:00")


st_kkn = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)

st_evn = client.get_waveforms(network="IO", station="EVN", location="*", channel="HHZ", starttime=start, endtime=end)

tr_kkn = st_kkn[0]
tr_evn = st_evn[0]

tr_kkn.filter("bandpass", freqmin=1.0, freqmax=10.0)
tr_evn.filter("bandpass", freqmin=1.0, freqmax=10.0)

fs_kkn = tr_kkn.stats.sampling_rate
fs_evn = tr_evn.stats.sampling_rate

sta_seconds = 1
lta_seconds = 10

# KKN
sta_kkn = int(sta_seconds * fs_kkn)
lta_kkn = int(lta_seconds * fs_kkn)

cft_kkn = classic_sta_lta(
    tr_kkn.data,
    sta_kkn,
    lta_kkn
)

# EVN
sta_evn = int(sta_seconds * fs_evn)
lta_evn = int(lta_seconds * fs_evn)

cft_evn = classic_sta_lta(
    tr_evn.data,
    sta_evn,
    lta_evn
)

# Triggers
triggers_kkn = trigger_onset(cft_kkn, 3.0, 0.5)
triggers_evn = trigger_onset(cft_evn, 3.0, 0.5)

print("KKN sampling rate:", fs_kkn)
print("EVN sampling rate:", fs_evn)

print("\nKKN triggers:")
for trigger in triggers_kkn:
    print(
        tr_kkn.stats.starttime +
        trigger[0] / fs_kkn
    )

print("\nEVN triggers:")
for trigger in triggers_evn:
    print(
        tr_evn.stats.starttime +
        trigger[0] / fs_evn
    )


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))

plt.plot(
    tr_kkn.times(),
    cft_kkn,
    label="KKN"
)

plt.plot(
    tr_evn.times(),
    cft_evn,
    label="EVN"
)

plt.axhline(
    3.0,
    linestyle="--",
    label="Trigger = 1.0"
)

plt.axhline(
    0.5,
    linestyle=":",
    label="Release = 0.5"
)

plt.xlabel("Seconds from 02:52:00 UTC")
plt.ylabel("STA/LTA")
plt.title("KKN vs EVN STA/LTA")
plt.legend()
plt.grid()

plt.show()