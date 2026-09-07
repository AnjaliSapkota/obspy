import numpy as np
from obspy import Trace, Stream, UTCDateTime

data = np.random.randn(500) * 0.1

t = np.linspace(0, 10, 500)

earthquake = np.sin(2 * np.pi * 3 * t) * 2

data[200:300] += earthquake[200:300]

tr = Trace(data = data)

tr.stats.station = "TEST"
tr.stats.network = "NP"
tr.stats.channel = "BHZ"

tr.stats.starttime = UTCDateTime("2026-09-04T12:00:00")

tr.stats.sampling_rate = 50

st = Stream([tr])

print(st)

print(tr.stats)
print(tr.stats.endtime)

st.plot()