from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime

client = Client("https://seiscomp.alertnepal.online")

stations = [
    {"sta": "KKN", "lat": 27.8000, "lon": 85.2790, "net": "NK", "cha": "BHZ", "loc": "*"},
    {"sta": "EVN", "lat": 27.95865, "lon": 86.811653, "net": "IO", "cha": "BHZ", "loc": "*"},
    {"sta": "EQM10", "lat": 28.299517, "lon": 83.960148, "net": "NP", "cha": "EHZ", "loc": "*"}
]

start = UTCDateTime("2026-09-22T07:45:00")
end = UTCDateTime("2026-09-22T07:55:00")

fs = 20.0
fmin = 1.0
fmax = 8.0

def waveform(station, start, end):
    stream = client.get_waveforms(network = station["net"], station = station["sta"], location = station["loc"], channel = station["cha"], starttime = start, endtime = end)
 
    if len(stream) == 0:
        raise RuntimeError("No waveform found") 
    stream.merge(method = 1, fill_value = "interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage = 0.05, type ="hann")
    stream.filter("bandpass", freqmin = fmin, freqmax = fmax, corners = 4, zerophase = True)
    stream.interpolate(sampling_rate = fs, method = "linear")
    return stream[0]

def fetch_all_traces(stations, start, end):
    traces = {}
    for st_info in stations:
        name = f"{st_info['net']}.{st_info['sta']}"
        try:
            tr = waveform(st_info, start, end)
            traces[name] = tr
        except Exception as e:
            print(f"Skipping {name}: {e}")
    return traces

def waveform_plot(traces):
    fig, axes = plt.subplots(3,1, figsize= (12,8))
    for ax, (name,trace) in zip(axes,traces.items()):
        data = trace.data.astype(float)
        sampling_rate = trace.stats.sampling_rate
        time = (np.arange(len(data)) / sampling_rate)
        ax.plot(time, data, linewidth = 0.8)
        ax.grid(True)

    fig.tight_layout()
    plt.show()

def spectogram(traces):
    fig, axes = plt.subplots(3,1, figsize = (12,8))
    for ax, (name,trace) in zip(axes,traces.items()):
        data = trace.data.astype(float)
        sampling_rate = trace.stats.sampling_rate
        time = (np.arange(len(data)) / sampling_rate)

        nperseg = min(512, len(data))
        noverlap = int(nperseg * 0.75 )
        ax.specgram(
            data,
            NFFT=nperseg,
            Fs=sampling_rate,
            noverlap=noverlap
        )
        ax.set_ylabel( f"{name}\nFrequency (Hz)" )
        ax.grid(True)
    plt.show()


traces = fetch_all_traces(stations, start, end)
waveform_plot(traces)
spectogram(traces)