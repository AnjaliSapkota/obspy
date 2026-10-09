import numpy as np


def get_three_components(
    client, station, start, end, fs, fmin, fmax, wait_sec
):
  """Fetches 3-component seismic waveforms, normalizes gain/response to m/s, and filters signals."""
  st = client.get_waveforms(
      network=station["net"],
      station=station["sta"],
      location=station["loc"],
      channel=station["cha"][:2] + "?",
      starttime=start - wait_sec,
      endtime=end,
  )

  if len(st) == 0:
    raise RuntimeError(f"No waveforms found for {station['sta']}")

  # Remove instrument response or apply raw digitizer count scaling
  try:
    inv = client.get_stations(
        network=station["net"],
        station=station["sta"],
        location=station["loc"],
        channel=station["cha"][:2] + "?",
        level="response",
    )
    st.remove_response(inventory=inv, output="VEL")
  except Exception:
    for tr in st:
      if np.max(np.abs(tr.data)) > 10.0:
        tr.data = tr.data.astype(float) / 1e8

  zs = st.select(component="Z")
  if len(zs) == 0:
    raise RuntimeError("Vertical component missing")

  wanted = station["cha"][:2]
  z_ref = next((t for t in zs if t.stats.channel[:2] == wanted), zs[0])
  loc_ref = z_ref.stats.location
  band_ref = z_ref.stats.channel[:2]

  st = st.select(location=loc_ref, channel=band_ref + "?")
  horiz = st.select(component="N") + st.select(component="E")

  if len(horiz) < 2:
    raise RuntimeError(
        f"Need two horizontal components for S-picking ({band_ref}*)"
    )
  st = st.select(component="Z") + horiz

  st.merge(method=1, fill_value="interpolate")
  st.detrend("linear")
  st.detrend("demean")
  st.taper(max_percentage=0.05, type="hann")
  st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
  st.interpolate(sampling_rate=fs, method="linear")

  t0 = max(tr.stats.starttime for tr in st)
  t1 = min(tr.stats.endtime for tr in st)
  st.trim(t0, t1)

  n = min(tr.stats.npts for tr in st)
  for tr in st:
    tr.data = tr.data[:n]

  z_list = st.select(component="Z")
  if len(z_list) == 0:
    raise RuntimeError("Vertical component missing")

  z = z_list[0]
  hors = [tr for tr in st if tr.stats.channel[-1] != "Z"]

  if len(hors) < 2:
    raise RuntimeError("Need two horizontal components for S-picking")

  # Vector envelope for picking
  hor_energy = np.zeros(n, dtype=float)
  for tr in hors:
    hor_energy += tr.data.astype(float) ** 2
  env = np.sqrt(hor_energy)

  # Component-wise maximum horizontal trace for ML calculation
  h_abs_data = [np.abs(tr.data.astype(float)) for tr in hors]
  max_hor_trace = np.maximum(h_abs_data[0], h_abs_data[1])

  return z, env, max_hor_trace