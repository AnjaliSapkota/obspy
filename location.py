import numpy as np
from scipy.optimize import least_squares


def latlon_to_xy(lat, lon, lat0, lon0):
  x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
  y = (lat - lat0) * 110.574
  return x, y


def xy_to_latlon(x, y, lat0, lon0):
  lat = lat0 + y / 110.574
  lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))
  return lat, lon


def locate_epicenter_lsq(obs, source_depth_km=0.0):
  """Solves for 2D epicenter using non-linear least squares trilateration."""
  n = len(obs)
  if n < 3:
    raise ValueError("Need at least 3 stations")

  lat0 = float(np.mean([o[0] for o in obs]))
  lon0 = float(np.mean([o[1] for o in obs]))

  sx, sy, sr = [], [], []
  for lat, lon, hypo in obs:
    r = (0.1 if hypo <= source_depth_km else np.sqrt(hypo**2 - source_depth_km**2))
    x, y = latlon_to_xy(lat, lon, lat0, lon0)
    sx.append(x)
    sy.append(y)
    sr.append(r)
  sx, sy, sr = np.array(sx), np.array(sy), np.array(sr)

  if len(set(zip(np.round(sx, 4), np.round(sy, 4)))) < n:
    raise ValueError("Two stations have identical coordinates")

  def resid(p):
    return np.hypot(p[0] - sx, p[1] - sy) - sr

  starts = [np.array([sx.mean(), sy.mean()])]

  A = np.column_stack([2 * (sx[1:] - sx[0]), 2 * (sy[1:] - sy[0])])
  b = (sr[0] ** 2 - sr[1:] ** 2) + (sx[1:] ** 2 - sx[0] ** 2) + (sy[1:] ** 2 - sy[0] ** 2)
  if np.linalg.matrix_rank(A) == 2:
    starts.append(np.linalg.lstsq(A, b, rcond=None)[0])

  spread = max(np.ptp(sx), np.ptp(sy), 10.0)
  for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
    starts.append(starts[0] + 0.5 * spread * np.array([np.cos(ang), np.sin(ang)]))

  best = None
  for s0 in starts:
    sol = least_squares(resid, s0, method="lm")
    if best is None or sol.cost < best.cost:
      best = sol

  x_epi, y_epi = best.x
  residuals = resid(best.x)
  rms = float(np.sqrt(np.mean(residuals**2)))
  lat_epi, lon_epi = xy_to_latlon(x_epi, y_epi, lat0, lon0)

  return float(lat_epi), float(lon_epi), rms, [float(r) for r in residuals]