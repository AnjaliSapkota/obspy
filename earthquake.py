import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy.clients.fdsn import Client


# ============================================================
# 1. SETTINGS
# ============================================================

start = obspy.UTCDateTime("2026-09-15T19:59:00")
end   = obspy.UTCDateTime("2026-09-15T20:31:00")

client = Client("https://seiscomp.alertnepal.online")

st_kkn = client.get_waveforms(
    network="NK",
    station="KKN",
    location="*",
    channel="BHZ",
    starttime=start,
    endtime=end
)

st_evn = client.get_waveforms(
    network="IO",
    station="EVN",
    location="*",
    channel="BHZ",
    starttime=start,
    endtime=end
)

st_eqm10 = client.get_waveforms(
    network="NP",
    station="EQM10",
    location="*",
    channel="HNZ",
    starttime=start,
    endtime=end
)



print("\nKKN:")
print(st_kkn)

print("\nEVN:")
print(st_evn)

print("\nEQM10:")
print(st_eqm10)


tr_kkn = st_kkn[0]
tr_evn = st_evn[0]
tr_eqm10 = st_eqm10[0]


print("TRACE INFORMATION")

print("\nKKN")
print("ID            :", tr_kkn.id)
print("Start time    :", tr_kkn.stats.starttime)
print("End time      :", tr_kkn.stats.endtime)
print("Sampling rate :", tr_kkn.stats.sampling_rate)
print("Samples       :", tr_kkn.stats.npts)

print("\nEVN")
print("ID            :", tr_evn.id)
print("Start time    :", tr_evn.stats.starttime)
print("End time      :", tr_evn.stats.endtime)
print("Sampling rate :", tr_evn.stats.sampling_rate)
print("Samples       :", tr_evn.stats.npts)

print("\nEQM10")
print("ID            :", tr_eqm10.id)
print("Start time    :", tr_eqm10.stats.starttime)
print("End time      :", tr_eqm10.stats.endtime)
print("Sampling rate :", tr_eqm10.stats.sampling_rate)
print("Samples       :", tr_eqm10.stats.npts)



time_kkn = tr_kkn.times("matplotlib")
time_evn = tr_evn.times("matplotlib")
time_eqm10 = tr_eqm10.times("matplotlib")




fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 10),
    sharex=True
)




axes[0].plot(
    time_kkn,
    tr_kkn.data,
    linewidth=0.5
)

axes[0].set_title("NK.KKN.BHZ — Raw Waveform")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)




axes[1].plot(
    time_evn,
    tr_evn.data,
    linewidth=0.5
)

axes[1].set_title("IO.EVN.BHZ — Raw Waveform")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)




axes[2].plot(
    time_eqm10,
    tr_eqm10.data,
    linewidth=0.5
)

axes[2].set_title("NP.EQM10.HNZ — Raw Waveform")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("UTC Time")
axes[2].grid(True, alpha=0.3)




axes[2].xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M:%S")
)

fig.autofmt_xdate()



plt.suptitle(
    "Three-Station Waveform Comparison — 2026-09-15",
    fontsize=16
)

plt.tight_layout()

plt.show()