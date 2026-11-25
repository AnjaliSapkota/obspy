
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# FDSN server
client = Client("https://seiscomp.alertnepal.online")

# Station information
network = "NP"
station = "EQM07"
location = "*"
channel = "EH*"

# Time range (UTC)
starttime = UTCDateTime("2026-11-24T01:00:00")
endtime = UTCDateTime("2026-11-24T02:00:00")

try:
    print("Downloading waveform data...")

    st = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=starttime,
        endtime=endtime
    )

    if len(st) == 0:
        print("No waveform data available.")
    else:
        # Save MiniSEED file
        filename = "NP_EQM07_20261124_0100_0200.mseed"

        st.write(filename, format="MSEED")

        print("\nDownload successful!")
        print("Saved:", filename)
        print("Number of traces:", len(st))
        print("Total samples:", sum(len(tr.data) for tr in st))

        print("\nWaveform details:")
        print(st)

except Exception as e:
    print("Download failed:", e)
