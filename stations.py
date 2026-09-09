import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("EARTHSCOPE")

t_start = UTCDateTime("2026-08-26T01:30:00")
t_end = UTCDateTime("2026-08-26T04:00:00")

st = client.get_waveforms(
    network="IO",
    station="EVN",
    location="*",
    channel="HHZ, BHZ",
    starttime=t_start,
    endtime=t_end,
)

st.merge(fill_value="latest")

print(st)