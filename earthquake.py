import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from location import locate_epicenter_lsq
from magnitude import calculate_epicentral_distance, calculate_ml
from pickers import pick_p, pick_s
from plotting import circle_map_png, spectrogram_png, waveform_stalta_png
from quality import calculate_quality_metrics
from waveforms import get_three_components


def run_location( base_url, stations, start, end, fs=20.0, fmin=1.0, fmax=8.0, wait_sec=30.0, vp=6.0, vs=3.5, sta_p=1.0, lta_p=10.0, thr_on=3.5, thr_off=1.5, sta_s=1.0, lta_s=8.0, min_sp_sec=4.0, max_sp_sec=120.0, thr_on_s=2.5, thr_off_s=1.0, exclude_rejected=True):
  client = Client(base_url)
  k = vp * vs / (vp - vs)

  station_results = []
  obs, names, used_entries = [], [], []

  for station in stations:
    name = f"{station['net']}.{station['sta']}"
    entry = {"name": name, "lat": station["lat"], "lon": station["lon"]}

    try:
      z, env, max_hor = get_three_components(client, station, start, end, fs, fmin, fmax, wait_sec)
      tp, cft_p, p_sta_time, p_trigger = pick_p(z, fs, start, sta_p, lta_p, thr_on, thr_off)

      if tp is None:
        entry["error"] = "No P trigger found"
        entry["quality"] = "REJECT"
        station_results.append(entry)
        continue

      ts, cft_s, s_sta_time, s_trigger = pick_s(env, fs, z.stats.starttime, tp, sta_s, lta_s, min_sp_sec, max_sp_sec, thr_on_s, thr_off_s)

      if ts is None:
        entry["error"] = (f"No S trigger found in window (P at {tp.isoformat()[11:23]})")
        entry["quality"] = "REJECT"
        entry["waveform_png"] = waveform_stalta_png(name, z, env, cft_p, cft_s, tp, None, p_sta_time, None, thr_on, thr_off, thr_on_s, thr_off_s)
        entry["spectrogram_png"] = spectrogram_png(name, z)
        station_results.append(entry)
        continue

      sp_diff = float(ts - tp)
      dist_km = float(sp_diff * k)
      q_info = calculate_quality_metrics(z, env, fs, tp, ts, p_trigger, s_trigger)

      entry.update({
          "p_time": tp.isoformat(),
          "s_time": ts.isoformat(),
          "sp_diff": sp_diff,
          "hypo_distance_km": dist_km,
          "_trace_starttime": z.stats.starttime,
          "_env": env,
          "_max_hor": max_hor,
          "waveform_png": waveform_stalta_png(name, z, env, cft_p, cft_s, tp, ts, p_sta_time, s_sta_time, thr_on, thr_off, thr_on_s, thr_off_s,),
          "spectrogram_png": spectrogram_png(name, z), **q_info,
      })

      if exclude_rejected and entry.get("quality") == "REJECT":
        entry.setdefault("quality_notes", []).append(
            "Excluded from location (REJECT quality)"
        )
      else:
        obs.append((station["lat"], station["lon"], dist_km))
        names.append(name)
        used_entries.append(entry)

    except Exception as e:
      entry["error"] = str(e)
      entry["quality"] = "REJECT"

    station_results.append(entry)

  def _clean_private_fields(results):
    for s in results:
      s.pop("_trace_starttime", None)
      s.pop("_env", None)
      s.pop("_max_hor", None)

  if len(obs) >= 3:
    try:
      lat_epi, lon_epi, rms, residuals = locate_epicenter_lsq(obs)
      origin_times, station_magnitudes = [], []

      for entry, res in zip(used_entries, residuals):
        entry["residual_km"] = res
        origin_times.append(
            UTCDateTime(entry["p_time"]) - entry["hypo_distance_km"] / vp
        )

        epi_dist = calculate_epicentral_distance(
            entry["lat"], entry["lon"], lat_epi, lon_epi
        )
        entry["epicentral_distance_km"] = epi_dist

        trace_max_hor = entry.get("_max_hor")
        trace_t0 = entry.get("_trace_starttime")

        if trace_max_hor is not None and trace_t0 is not None:
          s_idx = int((UTCDateTime(entry["s_time"]) - trace_t0) * fs)
          window_samples = int(2.0 * fs)
          s_win = trace_max_hor[
              max(0, s_idx) : min(
                  len(trace_max_hor), s_idx + window_samples
              )
          ]

          if len(s_win) > 0:
            peak_amp = float(np.max(s_win))
            peak_loc = np.argmax(s_win)
            peak_amp_nms = peak_amp * 1e9
            period = 0.2

            try:
              win_center = max(0, s_idx) + peak_loc
              half_w = int(0.5 * fs)
              sub_seg = trace_max_hor[
                  max(0, win_center - half_w) : min(
                      len(trace_max_hor), win_center + half_w
                  )
              ]
              zero_crossings = np.where(
                  np.diff(np.signbit(sub_seg - np.mean(sub_seg)))
              )[0]
              if len(zero_crossings) >= 2:
                period = float(
                    2.0 * (zero_crossings[1] - zero_crossings[0]) / fs
                )
                period = max(0.05, min(period, 2.0))
            except Exception:
              pass

            ml_val = calculate_ml(
                amplitude=peak_amp_nms,
                period=period,
                delta_km=epi_dist,
                station_correction=0.0,
            )
            entry["amplitude"] = peak_amp_nms
            entry["period"] = period
            entry["ml"] = round(ml_val, 2)
            station_magnitudes.append(ml_val)

      _clean_private_fields(station_results)
      avg_origin_time = UTCDateTime(
          float(np.mean([t.timestamp for t in origin_times]))
      ).isoformat()
      network_ml = (
          float(np.median(station_magnitudes)) if station_magnitudes else None
      )

      return {
          "located": True,
          "epicenter": {"lat": lat_epi, "lon": lon_epi},
          "magnitude_ml": (
              round(network_ml, 2) if network_ml is not None else None
          ),
          "rms_km": rms,
          "n_stations_used": len(obs),
          "origin_time": avg_origin_time,
          "circle_map_png": circle_map_png(obs, names, (lat_epi, lon_epi)),
          "stations": station_results,
      }

    except Exception as e:
      _clean_private_fields(station_results)
      return {
          "located": False,
          "error": f"Trilateration failed: {e}",
          "stations": station_results,
      }

  _clean_private_fields(station_results)
  return {
      "located": False,
      "error": f"Insufficient valid station picks ({len(obs)} usable, 3 required)",
      "stations": station_results,
  }