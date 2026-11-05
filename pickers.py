import numpy as np
from obspy.signal.trigger import recursive_sta_lta, trigger_onset


def aic_pick(trace_data, fs, approximate_idx, search_before=1.0, search_after=0.5, max_shift_sec=0.5):
  n = len(trace_data)
  i0 = max(1, approximate_idx - int(search_before * fs))
  i1 = min(n - 2, approximate_idx + int(search_after * fs))

  if i1 <= i0 + 10:
    return approximate_idx

  x = np.asarray(trace_data[i0:i1], dtype=float)
  aic = np.full(len(x), np.nan)

  for k in range(1, len(x) - 1):
    var1 = np.var(x[:k])
    var2 = np.var(x[k:])
    if var1 <= 0 or var2 <= 0:
      continue
    aic[k] = k * np.log10(var1) + (len(x) - k) * np.log10(var2)

  valid = np.isfinite(aic)
  if not np.any(valid):
    return approximate_idx

  local_idx = int(np.nanargmin(aic))

  # A minimum at the edge of the search window usually means the real onset
  # lies outside it.  Keep the STA trigger in that case.
  edge = max(2, int(0.05 * len(x)))
  if local_idx < edge or local_idx >= len(x) - edge:
    return approximate_idx

  pick_idx = i0 + local_idx
  if abs(pick_idx - approximate_idx) > max_shift_sec * fs:
    return approximate_idx

  return pick_idx

def pick_p(z, fs, start, sta=1.0, lta=10.0, thr_on=3.5, thr_off=1.5, aic_before=1.0, aic_after=0.5):
  nsta = int(sta * fs)
  nlta = int(lta * fs)
  cft = recursive_sta_lta(z.data, nsta, nlta)
  p_idx = None

  for on, _off in trigger_onset(cft, thr_on, thr_off):
    t = z.stats.starttime + on / fs
    if t >= start:
      p_idx = int(on)
      break

  if p_idx is None:
    return None, cft, None, None

  p_idx_aic = aic_pick(
      z.data, fs, p_idx, search_before=aic_before, search_after=aic_after,
      max_shift_sec=0.5,
  )
  p_time_sta = z.stats.starttime + p_idx / fs
  p_time_aic = z.stats.starttime + p_idx_aic / fs
  trigger_value = float(cft[p_idx])

  return p_time_aic, cft, p_time_sta, trigger_value


def pick_s(env, fs, t0, p_time, sta=1.0, lta=8.0, min_sp_sec=1.0, max_sp_sec=120.0, thr_on=2.5, thr_off=1.0, aic_before=1.5, aic_after=4.0):
  nsta = max(1, int(sta * fs))
  nlta = max(nsta + 1, int(lta * fs))
  cft = recursive_sta_lta(env, nsta, nlta)
  p_idx = int((p_time - t0) * fs)

  i0 = p_idx + int(min_sp_sec * fs)
  i1 = min(p_idx + int(max_sp_sec * fs), len(cft) - 1)

  if i0 >= i1 or i0 >= len(cft):
    return None, cft, None, None

  s_idx = None
  for on, off in trigger_onset(cft, thr_on, thr_off):
    if on < i0 <= off:
      s_idx = int(i0)
      break
    if i0 <= on <= i1:
      s_idx = int(on)
      break

  if s_idx is None:
    return None, cft, None, None

  s_idx_aic = aic_pick(
      env, fs, s_idx, search_before=aic_before, search_after=aic_after
  )
  s_time_sta = t0 + s_idx / fs
  s_time_aic = t0 + s_idx_aic / fs
  trigger_value = float(cft[s_idx])

  return s_time_aic, cft, s_time_sta, trigger_value