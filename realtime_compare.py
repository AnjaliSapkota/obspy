import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate as scipy_correlate, windows
from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

server = "ring.wscada.net:18000"
window = 10
max_delay = 2

stations = [
    {"net": "NP", "sta": "EQM10", "cha": "HNZ"},
    {"net": "NP", "sta": "EQM11", "cha": "HNZ"}
]

plt.ion()
plt.rcParams['keymap.quit'] = []
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# normalized cross correlation

def normalized_cross_correlation(vec1, vec2, fs, max_delay):
    # make both signal same length
    n = min(len(vec1), len(vec2))

    vec1 = np.asarray(vec1[:n], dtype = float)
    vec2 = np.asarray(vec2[:n], dtype=float)

    # remove offset
    vec1 = vec1 - np.mean(vec1)
    vec2 = vec2 - np.mean(vec2)

    # tukey taper
    taper = windows.tukey(n, alpha = 0.02)
    vec1 = vec1 * taper
    vec2 = vec2 * taper

    # calc full raw cross correlation
    # cc_full = scipy_correlate(vec1, vec2, mode = "full")

    # lag values
    lag_samples = np.arange(-(n-1), n)

    # only keep requested delay range
    max_shift = int(round(max_delay * fs))
    keep = np.abs(lag_samples) <= max_shift
    lag_samples = lag_samples[keep]
    # cc = cc_full[keep]

    # lag dependent normalization
    # for every lag corr/sqrt(ergy1*ergy2)
    # only overlapping samples are used
    normalized_cc = np.zeros(len(lag_samples))

    # calculate correlation for every lag
    for j, lag in enumerate(lag_samples):
        if lag < 0:
            #station 2 shifted earlier
            start1 = -lag
            end1 = n

            start2 = 0
            end2 = n + lag

        else:
            # station 2 shifted later
            start1 = 0
            end1 = n - lag

            start2 = lag
            end2 = n

        # extract overlapping parts
        x = vec1[start1:end1]
        y = vec2[start2:end2]

        if len(x) == 0 or len(y) == 0:
            continue

        # correlation for this overlap
        corr = np.sum(x * y)

        # energy of each signal
        energy1 = np.sum(x * x)
        energy2 = np.sum(y * y)

        # normalization
        if energy1 > 0 and energy2 > 0:
            normalized_cc[j] = (corr/np.sqrt(energy1 * energy2))

    # find maximum correlatio
    if len(normalized_cc) == 0:
        return 0, 0, [], []

    max_index = np.argmax(normalized_cc)
    correlation = normalized_cc[max_index]
    best_shift = lag_samples[max_index]
    lag = best_shift/fs

    return lag, correlation, lag_samples/fs, normalized_cc
     
class MyClient(EasySeedLinkClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = Stream()
        self.last_plot_time = 0.0 

    def on_data(self, trace):
        if trace is None or trace.stats.npts == 0:
            return

        try:
            # Maintain raw buffer
            self.stream += trace
            self.stream.merge(method=1, fill_value="interpolate")

            latest_time = max(tr.stats.endtime for tr in self.stream)
            self.stream.trim(starttime=latest_time - (window + 10))

            # Isolated processing copy (prevents Trace.stats.processing bloat)
            st = self.stream.copy()
            st.trim(starttime=latest_time - window, endtime=latest_time)

            # st.detrend("demean")
            # st.detrend("linear")
            # Filter narrower band (0.5 - 2.0 Hz) to eliminate uncorrelated high-frequency noise
            # st.filter("bandpass", freqmin=0.5, freqmax=5.0, zerophase=False)

        except Exception:
            return

        data10 = st.select(network="NP", station="EQM10", channel="HNZ")
        data11 = st.select(network="NP", station="EQM11", channel="HNZ")

        if not data10 or not data11:
            return

        tr10, tr11 = data10[0], data11[0]

        if (tr10.stats.endtime - tr10.stats.starttime < window * 0.8 or
            tr11.stats.endtime - tr11.stats.starttime < window * 0.8):
            return

        fs = tr10.stats.sampling_rate
        if fs != tr11.stats.sampling_rate:
            return

        common_start = max(tr10.stats.starttime, tr11.stats.starttime)
        common_end = min(tr10.stats.endtime, tr11.stats.endtime)

        if common_end <= common_start:
            return

        tr10_sync = tr10.slice(starttime=common_start, endtime=common_end)
        tr11_sync = tr11.slice(starttime=common_start, endtime=common_end)

        n = min(len(tr10_sync.data), len(tr11_sync.data))
        if n < 100:
            return

        vec10 = tr10_sync.data[:n]
        vec11 = tr11_sync.data[:n]

        # correlation
        lag, correlation, lags, cc = (normalized_cross_correlation(vec10, vec11, fs, max_delay))

        # norm10 = np.linalg.norm(vec10)
        # norm11 = np.linalg.norm(vec11)

        # if norm10 > 0 and norm11 > 0:
        #     max_shift = int(2 * fs)
        #     cc_full = scipy_correlate(vec10, vec11, mode="full") / (norm10 * norm11)
        #     mid = len(cc_full) // 2
            
        #     cc = cc_full[mid - max_shift : mid + max_shift + 1]
        #     max_idx = np.argmax(cc)
        #     correlation = cc[max_idx]
        #     shift = max_idx - max_shift
        #     lag = shift / fs

            # Extracted Timestamps
        utc_time_str = common_end.strftime("%H:%M:%S.%f")[:-4]
        sys_time_str = time.strftime("%H:%M:%S")

        print(
            f"[{sys_time_str} Sys | {utc_time_str} UTC] "
            f"EQM10 vs EQM11 | Lag: {lag:6.3f} s | Cross-Corr: {correlation:6.3f}"
        )


        # GUI Update Throttling (Limits render calls to once every 0.5s)
        current_time = time.time()
        if current_time - self.last_plot_time < 0.5:
            return
        self.last_plot_time = current_time

        # Render subplots
        for row, target in enumerate(stations):
            ax_time, ax_freq = axes[row][0], axes[row][1]
            ax_time.clear()
            ax_freq.clear()

            matched = st.select(
                network=target["net"],
                station=target["sta"],
                channel=target["cha"]
            )

            if not matched:
                continue

            tr_plot = matched[0]

            # waveforms
            ax_time.plot(tr_plot.times(), tr_plot.data)
            ax_time.set_title(f"{target['net']}.{target['sta']}.{target['cha']}")
            ax_time.set_ylabel("Amplitude")
            ax_time.set_xlabel("Time (s)")

            # PSD
            if len(tr_plot.data) > 20:
                freqs, psd = welch(tr_plot.data, fs=fs, nperseg=min(1024, len(tr_plot.data)))
                ax_freq.semilogy(freqs, psd)
                ax_freq.set_title(f"PSD — {target['net']}.{target['sta']}")
                ax_freq.set_xlabel("Frequency (Hz)")
                ax_freq.set_ylabel("PSD")

        # refresh GUI
        try:
            plt.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass


client = MyClient(server, autoconnect=False)
client.conn.timeout = 10

try:
    client.connect()
    for target in stations:
        client.select_stream(target["net"], target["sta"], target["cha"])

    print("Receiving live data...")
    print("Press Ctrl+C to stop.")
    client.run()

except KeyboardInterrupt:
    print("\nStream stopped by user.")
except Exception as e:
    print(f"\nConnection Error: {e}")
finally:
    client.close()
    plt.close("all")