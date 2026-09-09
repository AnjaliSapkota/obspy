# using client (standard obspy seedlink), pass host and port as positional parqameters 

# from obspy.clients.seedlink  import Client

# try:
#     client = Client("ring.wscada.net", 18000)
#     print("connected")

#     info = client.get_info('STATIONS')
#     print(info)

# except Exception as e:
#     print(f"error: {e}")


# using easyseedlinkclient, subclassing easyseedlinkclient - for live streaming 

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class FloodMonitor(EasySeedLinkClient):
    def __init__(self, server_url, autoconnect=True):
        super().__init__(server_url, autoconnect)
        self.stream = Stream()

        plt.ion()
        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [1, 2]}
        )
        self.fig.canvas.manager.set_window_title("Live Seismic & Spectrogram Monitor - Flood Signal Analysis")
        self.quadmesh = None

    def on_data(self, trace):
        # print(f"Recieved packet: {trace.id} | Samples: {len(trace)} | Time: {trace.stats.starttime}")
        self.stream = self.stream + trace
        self.stream.merge(method=1, fill_value='interpolate')

        endtime = self.stream[-1].stats.endtime     # latest time
        self.stream.trim(starttime = endtime - 40)      # keep only 30 seconds, maintain sliding window

        raw_tr = self.stream[0]
        duration = raw_tr.stats.endtime - raw_tr.stats.starttime

        if duration < 10.0:
            print(f"Buffering data... ({duration:.1f}s / 10.0s minimum)", end='\r')
            return

        tr = raw_tr.copy()

        # Keep raw stream buffer clean of accumulated metadata history
        self.stream[0].stats.processing = []

        tr.detrend("linear")
        tr.taper(max_percentage = 0.05)

        tr.filter("bandpass", freqmin = 0.5, freqmax = 20)

        data = tr.data
        sampling_rate = tr.stats.sampling_rate

        # amplitude indicators
        rms_amp = float(np.sqrt(np.mean(data ** 2)))
        peak_amp = float(np.max(np.abs(data)))

        # spectral band energies via FFT
        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), d = tr.stats.delta)

        # energy in key hydrological bands
        e_1_3 = float(np.sum(fft_vals[(freqs >= 1.0) & (freqs <= 3.0)]))   # Boulder / Bedload motion
        e_3_10 = float(np.sum(fft_vals[(freqs >= 3.0) & (freqs <= 10.0)])) # Water turbulence
        e_10_20 = float(np.sum(fft_vals[(freqs >= 10.0) & (freqs <= 20.0)])) # Fine sediment / Rain impact

        # Ratio of water turbulence to bedload movement
        wb_ratio = e_3_10 / (e_1_3 + 1e-6)

        timestamp_str = tr.stats.endtime.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        station_id = tr.id

        # 5. Extract signal data
        data = tr.data
        fs = tr.stats.sampling_rate
        time_vec = np.linspace(0, duration, len(data))

        # 6. Compute Spectrogram via SciPy
        # nperseg defines time-frequency resolution trade-off (~2 sec window)
        nperseg = int(min(fs * 2, len(data)))
        freqs, times, Sxx = signal.spectrogram(
            data, fs=fs, window='hann', nperseg=nperseg, noverlap=int(nperseg * 0.8)
        )

        # Convert power spectrum to decibels (dB) for enhanced contrast
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # --- Update Matplotlib Plots ---
        self.ax_wave.clear()
        self.ax_spec.clear()

        # Top Plot: Filtered Waveform
        self.ax_wave.plot(time_vec, data, color='#1f77b4', lw=0.8)
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_title(f"Live Trace: {tr.id} | Endtime: {tr.stats.endtime.strftime('%H:%M:%S')}")
        self.ax_wave.grid(True, linestyle='--', alpha=0.5)

        # Bottom Plot: Spectrogram
        pcm = self.ax_spec.pcolormesh(
            times, freqs, Sxx_db, shading='gouraud', cmap='viridis'
        )
        self.ax_spec.set_ylim(0.5, 20.0)  # Focus on key flood frequencies (0.5 - 20 Hz)
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_xlabel("Time Buffer (seconds)")
        self.ax_spec.grid(True, linestyle='--', alpha=0.3)

        # Annotate Key Hydrological Bands
        self.ax_spec.axhline(1.0, color='r', linestyle=':', alpha=0.7, label='Bedload (1-3 Hz)')
        self.ax_spec.axhline(3.0, color='r', linestyle=':', alpha=0.7)
        self.ax_spec.axhline(10.0, color='g', linestyle=':', alpha=0.7, label='Turbulence (3-10 Hz)')

        # Redraw window without blocking SeedLink socket thread
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
try:
    client = FloodMonitor("ring.wscada.net:18000", autoconnect=False)

    client.conn.timeout = 10

    client.connect()

    client.select_stream("NP", "EQM22", "HNZ")

    client.run()

except Exception as e:
    print(f"Stream error: {e}")