import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows
from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

SERVER = "ring.wscada.net:18000"
WINDOW = 10

stations = [
    {"net": "NP", "sta": "EQM11", "cha": "HNZ"},
    {"net": "NP", "sta": "EQM13", "cha": "HNZ"}
]

plt.ion()
plt.rcParams["keymap.quit"] = []

fig, axes = plt.subplots(2, 2, figsize=(12, 7))


def next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def cross_correlation(s1, s2, fs, max_delay):

    if fs <= 0:
        return None, None

    n1 = len(s1)
    n2 = len(s2)

    duration = min(n1, n2) / fs
    dt = 1.0 / fs

    if max_delay < 0:
        return None, None

    if max_delay > duration:
        max_delay = duration - dt

    max_lag_samples = int(np.round(max_delay * fs))

    v1 = s1.copy() - np.mean(s1)
    v2 = s2.copy() - np.mean(s2)

    taper = windows.tukey(len(v1), alpha=0.02)

    v1 *= taper
    v2 *= taper

    n_conv = n1 + n2 - 1
    n_fft = next_pow2(n_conv)

    x = np.zeros(n_fft)
    y = np.zeros(n_fft)

    x[:n1] = v1
    y[n1 - 1:n1 - 1 + n2] = v2

    F1 = np.fft.rfft(x)
    F2 = np.fft.rfft(y)

    raw_cc = np.fft.irfft(
        F1 * np.conj(F2),
        n=n_fft
    )


    lags_samples = np.arange(
        -max_lag_samples,
        max_lag_samples + 1
    )

    cc = np.zeros(len(lags_samples))

    for i, lag in enumerate(lags_samples):

        index = n1 - 1 + lag

        if 0 <= index < len(raw_cc):
            cc[i] = raw_cc[index]

    energy1 = np.insert(
        np.cumsum(v1 ** 2),
        0,
        0
    )

    energy2 = np.insert(
        np.cumsum(v2 ** 2),
        0,
        0
    )

    normalized_cc = np.zeros(len(lags_samples))

    for i, lag in enumerate(lags_samples):

        if lag >= 0:

            overlap = min(
                n1 - lag,
                n2
            )

            if overlap <= 0:
                continue

            e1 = (
                energy1[lag + overlap]
                - energy1[lag]
            )

            e2 = (
                energy2[overlap]
                - energy2[0]
            )

        else:

            shift = -lag

            overlap = min(
                n1,
                n2 - shift
            )

            if overlap <= 0:
                continue

            e1 = (
                energy1[overlap]
                - energy1[0]
            )

            e2 = (
                energy2[shift + overlap]
                - energy2[shift]
            )

        if e1 > 0 and e2 > 0:

            normalized_cc[i] = (
                cc[i]
                / np.sqrt(e1 * e2)
            )

    peak_index = np.argmax(normalized_cc)

    max_corr = normalized_cc[peak_index]

    lag = (
        lags_samples[peak_index]
        / fs
    )

    return max_corr, lag


class MyClient(EasySeedLinkClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream = Stream()
        self.last_plot_time = 0.0

    def on_data(self, trace):

        if trace is None or trace.stats.npts == 0:
            return

        try:
            self.stream += trace

            self.stream.merge(
                method=1,
                fill_value="interpolate"
            )

            latest_time = max(
                tr.stats.endtime
                for tr in self.stream
            )

            self.stream.trim(
                starttime=latest_time - (WINDOW + 10)
            )

            st = self.stream.copy()

            st.trim(
                starttime=latest_time - WINDOW,
                endtime=latest_time
            )

        except Exception:
            return

        data11 = st.select(
            network="NP",
            station="EQM11",
            channel="HNZ"
        )

        data13 = st.select(
            network="NP",
            station="EQM13",
            channel="HNZ"
        )

        if not data11 or not data13:
            return

        tr11 = data11[0]
        tr13 = data13[0]

        duration11 = (
            tr11.stats.endtime -
            tr11.stats.starttime
        )

        duration13 = (
            tr13.stats.endtime -
            tr13.stats.starttime
        )

        if duration11 < WINDOW * 0.8:
            return

        if duration13 < WINDOW * 0.8:
            return

        fs = tr11.stats.sampling_rate

        if fs != tr13.stats.sampling_rate:
            return

        if fs <= 0:
            return

        common_start = max(
            tr11.stats.starttime,
            tr13.stats.starttime
        )

        common_end = min(
            tr11.stats.endtime,
            tr13.stats.endtime
        )

        if common_end <= common_start:
            return

        tr11_sync = tr11.slice(
            starttime=common_start,
            endtime=common_end
        )

        tr13_sync = tr13.slice(
            starttime=common_start,
            endtime=common_end
        )

        n = min(
            len(tr11_sync.data),
            len(tr13_sync.data)
        )

        if n < 100:
            return

        vec11 = tr11_sync.data[:n]
        vec13 = tr13_sync.data[:n]

        correlation, lag = cross_correlation(
            vec11,
            vec13,
            fs=fs,
            max_delay=2.0
        )

        if correlation is not None:

            utc_time = common_end.strftime(
                "%H:%M:%S.%f"
            )[:-4]

            sys_time = time.strftime(
                "%H:%M:%S"
            )

            window_used = n / fs

            print(
                f"[{sys_time} Sys | {utc_time} UTC] "
                f"EQM11 vs EQM13 | "
                f"Window: {window_used:.2f} s | "
                f"Fs: {fs:.1f} Hz | "
                f"Lag: {lag:+.3f} s | "
                f"Cross-Corr: {correlation:.3f}"
            )

        current_time = time.time()

        if current_time - self.last_plot_time < 0.5:
            return

        self.last_plot_time = current_time

        for row, target in enumerate(stations):

            ax_time = axes[row][0]
            ax_freq = axes[row][1]

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

            ax_time.plot(
                tr_plot.times(),
                tr_plot.data
            )

            ax_time.set_title(
                f"{target['net']}.{target['sta']}.{target['cha']}"
            )

            ax_time.set_ylabel("Amplitude")
            ax_time.set_xlabel("Time (s)")

            freqs, psd = welch(
                tr_plot.data,
                fs=fs,
                nperseg=min(
                    1024,
                    len(tr_plot.data)
                )
            )

            ax_freq.semilogy(
                freqs,
                psd
            )

            ax_freq.set_title(
                f"PSD — {target['net']}.{target['sta']}"
            )

            ax_freq.set_xlabel("Frequency (Hz)")
            ax_freq.set_ylabel("PSD")

        try:
            plt.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception:
            pass


client = MyClient(
    SERVER,
    autoconnect=False
)

client.conn.timeout = 10

try:

    print("Connecting...")

    client.connect()

    print("Connected!")

    for target in stations:

        client.select_stream(
            target["net"],
            target["sta"],
            target["cha"]
        )

    print("Receiving EQM11 and EQM13 live data...")
    print("Waiting for a full 10-second window...")
    print("Press Ctrl+C to stop.")

    client.run()

except KeyboardInterrupt:

    print("\nStream stopped by user.")

except Exception as e:

    print(f"\nConnection Error: {e}")

finally:

    client.close()
    plt.close("all")