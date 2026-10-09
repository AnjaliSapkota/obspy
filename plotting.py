import base64
import io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_png_b64(fig):
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=130)
  plt.close(fig)
  return base64.b64encode(buf.getvalue()).decode()

def waveform_stalta_png( name,z, env, cft_p, cft_s, tp, ts, p_sta, s_sta, thr_on, thr_off, thr_on_s, thr_off_s,):
  t = z.times("matplotlib")
  fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

  axes[0].plot(t, z.data, color="black", lw=0.7, label="Vertical (Z)")
  axes[0].axvline(
      tp.matplotlib_date, color="#2f6fed", ls="--", lw=1.2, label="P Pick"
  )
  if ts is not None:
    axes[0].axvline(
        ts.matplotlib_date, color="#d1432b", ls="--", lw=1.2, label="S Pick"
    )
  axes[0].set_ylabel("Z Amplitude")
  axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
  axes[0].set_title(f"{name} — Waveform & Triggers")

  axes[1].plot(t, env, color="#555555", lw=0.7, label="Horizontal Envelope")
  axes[1].axvline(tp.matplotlib_date, color="#2f6fed", ls="--", lw=1.2)
  if ts is not None:
    axes[1].axvline(
        ts.matplotlib_date, color="#d1432b", ls="--", lw=1.2, label="S Pick"
    )
  axes[1].set_ylabel("Horiz. Env")
  axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)

  axes[2].plot(t, cft_p, color="#2f6fed", lw=0.8, label="P STA/LTA")
  axes[2].axhline(thr_on, color="#1f9e6b", ls=":", lw=1, label=f"On ({thr_on})")
  axes[2].axhline(thr_off, color="#888888", ls=":", lw=1, label=f"Off ({thr_off})")
  axes[2].axvline(tp.matplotlib_date, color="#2f6fed", ls="--", lw=1)
  if p_sta is not None:
    axes[2].axvline(
        p_sta.matplotlib_date, color="#e67e22", ls=":", lw=1.2, label="P Trigger"
    )
  axes[2].set_ylabel("STA/LTA (P)")
  axes[2].legend(loc="upper right", fontsize=8, framealpha=0.9)

  axes[3].plot(t, cft_s, color="#d1432b", lw=0.8, label="S STA/LTA")
  axes[3].axhline(
      thr_on_s, color="#1f9e6b", ls=":", lw=1, label=f"On ({thr_on_s})"
  )
  axes[3].axhline(
      thr_off_s, color="#888888", ls=":", lw=1, label=f"Off ({thr_off_s})"
  )
  axes[3].axvline(
      tp.matplotlib_date, color="#2f6fed", ls="--", alpha=0.4, lw=1
  )
  if ts is not None:
    axes[3].axvline(ts.matplotlib_date, color="#d1432b", ls="--", lw=1)
  if s_sta is not None:
    axes[3].axvline(
        s_sta.matplotlib_date, color="#e67e22", ls=":", lw=1.2, label="S Trigger"
    )
  axes[3].set_ylabel("STA/LTA (S)")
  axes[3].legend(loc="upper right", fontsize=8, framealpha=0.9)

  axes[3].xaxis_date()
  axes[3].set_xlabel("Time (UTC)")

  xlim = (
      z.stats.starttime.matplotlib_date,
      z.stats.endtime.matplotlib_date,
  )

  for ax in axes:
    ax.set_xlim(*xlim)
    ax.grid(True, linestyle=":", alpha=0.5)

  fig.tight_layout()
  return _fig_to_png_b64(fig)


def spectrogram_png(name, z):
  fig, ax = plt.subplots(figsize=(10, 3.2))
  ax.specgram(z.data, Fs=z.stats.sampling_rate, cmap="viridis")
  ax.set_ylabel("Frequency (Hz)")
  ax.set_xlabel("Time (s, from trace start)")
  ax.set_title(f"{name} — spectrogram")
  fig.tight_layout()
  return _fig_to_png_b64(fig)


def circle_map_png(obs, names, epicenter):
  fig, ax = plt.subplots(figsize=(7, 7))
  ang = np.linspace(0, 2 * np.pi, 360)

  for (lat, lon, d), nm in zip(obs, names):
    lat_circle = lat + (d / 110.574) * np.sin(ang)
    lon_circle = lon + (d / (111.32 * np.cos(np.radians(lat)))) * np.cos(ang)

    ax.plot(lon_circle, lat_circle, lw=1.2, label=f"{nm} ({d:.1f} km)")
    ax.plot(lon, lat, "^", color="k", ms=9)
    ax.annotate(nm, (lon, lat), textcoords="offset points", xytext=(6, 6))

  ax.plot(epicenter[1],epicenter[0],"r*",ms=18,label=f"Epicenter ({epicenter[0]:.3f}N, {epicenter[1]:.3f}E)",) 

  ax.set_aspect(1 / np.cos(np.radians(epicenter[0])))
  ax.set_xlabel("Longitude (°E)")
  ax.set_ylabel("Latitude (°N)")
  ax.grid(True)
  ax.legend(loc="upper right", fontsize=8)
  ax.set_title("Trilateration from S-P travel times")

  fig.tight_layout()
  return _fig_to_png_b64(fig)