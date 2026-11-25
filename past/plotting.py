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


def _place_station_labels(ax, fig, obs, names, epicenter):
  """Place station names where they do not cover markers, legend or each other."""
  from matplotlib.transforms import Bbox

  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  axes_box = ax.get_window_extent(renderer=renderer)
  obstacles = []
  for lat, lon, _distance in obs:
    x, y = ax.transData.transform((lon, lat))
    obstacles.append(Bbox.from_bounds(x - 8, y - 8, 16, 16))
  x, y = ax.transData.transform((epicenter[1], epicenter[0]))
  obstacles.append(Bbox.from_bounds(x - 12, y - 12, 24, 24))
  legend = ax.get_legend()
  if legend is not None:
    obstacles.append(legend.get_window_extent(renderer=renderer))

  candidates = (
      (9, 9, "left", "bottom"),
      (9, -9, "left", "top"),
      (-9, 9, "right", "bottom"),
      (-9, -9, "right", "top"),
      (11, 0, "left", "center"),
      (-11, 0, "right", "center"),
      (0, 12, "center", "bottom"),
      (0, -12, "center", "top"),
      (24, 16, "left", "bottom"),
      (-24, 16, "right", "bottom"),
      (24, -16, "left", "top"),
      (-24, -16, "right", "top"),
      (0, 24, "center", "bottom"),
      (0, -24, "center", "top"),
  )
  style = dict(
      textcoords="offset points", fontsize=8,
      bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
  )

  def outside(box):
    return (
        max(0.0, axes_box.x0 - box.x0) + max(0.0, box.x1 - axes_box.x1)
        + max(0.0, axes_box.y0 - box.y0) + max(0.0, box.y1 - axes_box.y1)
    )

  for index in sorted(range(len(obs)), key=lambda i: -obs[i][0]):
    lat, lon, _distance = obs[index]
    best = None
    best_score = None
    for dx, dy, horizontal, vertical in candidates:
      trial = ax.annotate(
          names[index], (lon, lat), xytext=(dx, dy),
          ha=horizontal, va=vertical, **style,
      )
      box = trial.get_window_extent(renderer=renderer)
      padded = Bbox.from_extents(box.x0 - 2, box.y0 - 2, box.x1 + 2, box.y1 + 2)
      score = (
          sum(1 for item in obstacles if padded.overlaps(item)),
          outside(padded),
      )
      if score == (0, 0):
        if best is not None:
          best.remove()
        best = trial
        best_score = score
        break
      if best_score is None or score < best_score:
        if best is not None:
          best.remove()
        best = trial
        best_score = score
      else:
        trial.remove()
    if best is None:
      best = ax.annotate(
          names[index], (lon, lat), xytext=(9, 9),
          ha="left", va="bottom", **style,
      )
    obstacles.append(best.get_window_extent(renderer=renderer))


def circle_map_png(
    obs, names, epicenter, title="Trilateration from S-P travel times",
    legend_labels=None,
):
  fig, ax = plt.subplots(figsize=(7, 7))
  ang = np.linspace(0, 2 * np.pi, 360)

  for index, ((lat, lon, d), nm) in enumerate(zip(obs, names)):
    lat_circle = lat + (d / 110.574) * np.sin(ang)
    lon_circle = lon + (d / (111.32 * np.cos(np.radians(lat)))) * np.cos(ang)

    legend_label = (
        legend_labels[index] if legend_labels is not None
        else f"{nm} ({d:.1f} km)"
    )
    ax.plot(lon_circle, lat_circle, lw=1.2, label=legend_label)
    ax.plot(lon, lat, "^", color="k", ms=9)

  ax.plot(
      epicenter[1], epicenter[0], "r*", ms=18,
      label=f"Epicenter ({epicenter[0]:.3f}N, {epicenter[1]:.3f}E)",
  )

  ax.set_aspect(1 / np.cos(np.radians(epicenter[0])))
  ax.set_xlabel("Longitude (°E)")
  ax.set_ylabel("Latitude (°N)")
  ax.grid(True)
  ax.legend(loc="upper right", fontsize=8)
  ax.set_title(title)
  fig.tight_layout()

  _place_station_labels(ax, fig, obs, names, epicenter)
  return _fig_to_png_b64(fig)
