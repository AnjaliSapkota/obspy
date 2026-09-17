from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt

client = Client("https://seiscomp.alertnepal.online")

# Same time period for both stations
start = UTCDateTime("2026-09-15T00:00:00")
end   = UTCDateTime("2026-09-15T00:58:00")

st23 = client.get_waveforms("IO", "EVN", "*", "BHZ", start, end)
st24 = client.get_waveforms("NK", "KKN", "*", "BHZ", start, end)

tr23 = st23[0].copy()
tr24 = st24[0].copy()

# Remove mean
tr23.detrend("demean").detrend("linear")
tr24.detrend("demean").detrend("linear")

# tr23.filter("bandpass", freqmin=0.1, freqmax=8.0, zerophase=True)
# tr24.filter("bandpass", freqmin=0.1, freqmax=8.0, zerophase=True)

# tr24.resample(20)

# Pre-filter prevents noise amplification during deconvolution
try:
    inv23 = client.get_stations(network="IO", station="EVN", channel="BHZ", 
                                starttime=start, endtime=end, level="response")
    inv24 = client.get_stations(network="NK", station="KKN", channel="BHZ", 
                                starttime=start, endtime=end, level="response")

    pre_filt = (0.05, 0.1, 8.0, 10.0)
    
    # Remove instrument response using the inventory objects
    tr23.remove_response(inventory=inv23, output="VEL", pre_filt=pre_filt, zerophase=True)
    tr24.remove_response(inventory=inv24, output="VEL", pre_filt=pre_filt, zerophase=True)
    unit_label = "Velocity (m/s)"
    
except Exception as e:
    print(f"Response removal failed: {e}. Falling back to normalization.")
    
    # Apply bandpass to prevent baseline drift
    tr23.filter("bandpass", freqmin=0.1, freqmax=8.0, zerophase=True)
    tr24.filter("bandpass", freqmin=0.1, freqmax=8.0, zerophase=True)
    
    # Peak-normalization
    tr23.data = tr23.data / abs(tr23.data).max()
    tr24.data = tr24.data / abs(tr24.data).max()
    unit_label = "Normalized Amplitude"

print(tr23)
print(tr24)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(tr23.times(), tr23.data, color="tab:blue", lw=0.8, label="IO.EVN")
ax1.set_ylabel(unit_label)
ax1.set_title("EVN vs KKN Waveform Comparison")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right")

ax2.plot(tr24.times(), tr24.data, color="tab:orange", lw=0.8, label="NK.KKN")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel(unit_label)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()