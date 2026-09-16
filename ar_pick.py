from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import ar_pick

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:52:00")
end   = UTCDateTime("2026-08-26T02:53:00")

# KKN
kkn_z = client.get_waveforms(network="NK", station="KKN",location="*", channel="BHZ",starttime=start, endtime=end)[0]

kkn_n = client.get_waveforms(network="NK", station="KKN",location="*", channel="BHN",starttime=start, endtime=end)[0]

kkn_e = client.get_waveforms(network="NK", station="KKN",location="*", channel="BHE",starttime=start, endtime=end)[0]

# EVN
evn_z = client.get_waveforms(network="IO", station="EVN",location="*", channel="BHZ",starttime=start, endtime=end)[0]

evn_n = client.get_waveforms(network="IO", station="EVN",location="*", channel="BHN",starttime=start, endtime=end)[0]

evn_e = client.get_waveforms(
    network="IO", station="EVN",
    location="*", channel="BHE",
    starttime=start, endtime=end
)[0]

for tr in [kkn_z, kkn_n, kkn_e,evn_z, evn_n, evn_e]:
    tr.detrend("demean")
    tr.detrend("linear")

fs_kkn = kkn_e.stats.sampling_rate
fs_evn = evn_e.stats.sampling_rate

print("kkn",fs_kkn)
print("evn",fs_evn)


p_kkn, s_kkn = ar_pick(
    kkn_z.data,
    kkn_n.data,
    kkn_e.data,
    fs_kkn,
    f1=1.0, f2=8.0,       # Bandpass frequency range (Hz)
    lta_p=5.0, sta_p=0.4,  # LTA/STA lengths for P (seconds)
    lta_s=5.0, sta_s=0.4,  # LTA/STA lengths for S (seconds)
    m_p=3, m_s=6,          # Model order parameters
    l_p=0.2, l_s=0.4,      # Variance window lengths
    s_pick=True            # Enable S pick
)

print("KKN P:", p_kkn)
print("KKN S:", s_kkn)

# evn
p_evn, s_evn = ar_pick(
    evn_z.data,
    evn_n.data,
    evn_e.data,
    fs_evn,
    f1=1.0, f2=8.0,       # Bandpass frequency range (Hz)
    lta_p=5.0, sta_p=0.4,  # LTA/STA lengths for P (seconds)
    lta_s=5.0, sta_s=0.4,  # LTA/STA lengths for S (seconds)
    m_p=3, m_s=6,          # Model order parameters
    l_p=0.2, l_s=0.4,      # Variance window lengths
    s_pick=True            # Enable S pick
)

print(" EVN P:", p_evn)
print("EVN S:", s_evn)

import matplotlib.pyplot as plt

time_kkn = kkn_z.times()

plt.figure(figsize=(12, 6))

plt.plot(time_kkn, kkn_z.data, label="BHZ")

# plt.axvline(
#     p_kkn,
#     linestyle="--",
#     label=f"P pick = {p_kkn:.2f} s"
# )

plt.xlabel("Seconds after 02:52:00 UTC")
plt.ylabel("Amplitude")
plt.title("KKN - P-wave pick")
plt.legend()
plt.grid()

plt.show()


time_evn = evn_z.times()

plt.figure(figsize=(12, 6))

plt.plot(time_evn, evn_z.data, label="BHZ")

# plt.axvline(
#     p_evn,
#     linestyle="--",
#     label=f"P pick = {p_evn:.2f} s"
# )

plt.xlabel("Seconds after 02:52:00 UTC")
plt.ylabel("Amplitude")
plt.title("evN - P-wave pick")
plt.legend()
plt.grid()

plt.show()