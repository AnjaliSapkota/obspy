import numpy as np
from obspy import Stream


def _merge_block(traces):
    block = Stream()

    for trace in traces:
        block += trace

    if len(block) > 1:
        block.merge(method=1, fill_value="interpolate")

    return block


def merge_with_max_gap(stream, max_gap_sec=0.5):
    if stream is None or len(stream) == 0:
        return Stream()

    max_gap_sec = float(max_gap_sec)

    if max_gap_sec < 0:
        raise ValueError("Maximum gap must not be negative")

    segments = [
        trace
        for trace in stream.copy().split()
        if trace.stats.npts > 0
    ]

    groups = {}

    for trace in segments:
        key = (
            trace.stats.network,
            trace.stats.station,
            trace.stats.location,
            trace.stats.channel,
        )
        groups.setdefault(key, []).append(trace)

    merged = Stream()

    for traces in groups.values():
        traces.sort(key=lambda trace: (trace.stats.starttime, trace.stats.endtime))

        block = []

        for trace in traces:
            if not block:
                block.append(trace)
                continue

            previous = block[-1]

            gap = (
                trace.stats.starttime
                - previous.stats.endtime
                - float(previous.stats.delta)
            )

            if gap <= max_gap_sec + 1e-9:
                block.append(trace)
            else:
                merged += _merge_block(block)
                block = [trace]

        if block:
            merged += _merge_block(block)

    return merged


def prepare_stream(stream, fs, fmin, fmax, max_gap_sec=0.5):
    
    if stream is None or len(stream) == 0:
        raise ValueError("No data in stream")

    fs = float(fs)
    fmin = float(fmin)
    fmax = float(fmax)

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Invalid bandpass frequency settings")

    if fmax >= fs / 2:
        raise ValueError("Maximum filter frequency must be below Nyquist frequency")
    

    # Merge only across small gaps; keep larger gaps as separate traces.
    st = merge_with_max_gap(stream, max_gap_sec)

    # Remove traces with fewer than 20 samples.
    for trace in list(st):
        if trace.stats.npts < 20:
            st.remove(trace)

    if len(st) == 0:
        raise ValueError("No usable traces after merging")

    # Validate the filter band against each trace's real sampling rate. The
    # configured fs above is only a target; stations may deliver other rates.
    for trace in st:
        nyquist = float(trace.stats.sampling_rate) / 2.0

        if fmax >= nyquist:
            raise ValueError(
                f"Maximum filter frequency {fmax} Hz must be below "
                f"Nyquist {nyquist} Hz for {trace.id}"
            )

    # Remove linear trend and mean.
    st.detrend("linear")
    st.detrend("demean")

    # Apply 2% left-sided cosine taper.
    st.taper(max_percentage=0.02, max_length=2.0, type="cosine", side="left",)

    # Apply bandpass filter.
    try:
        st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True,)
    except Exception as error:
        raise ValueError(f"Bandpass filtering failed: {error}") from error

    # Resample traces to the configured sampling frequency.
    for trace in list(st):

        current_fs = float(trace.stats.sampling_rate)

        if abs(current_fs - fs) > 1e-6:
            try:
                trace.resample(fs)

            except Exception as error:
                raise ValueError(f"Resampling failed for {trace.id}: {error}") from error

    if len(st) == 0:
        raise ValueError("No usable traces after resampling")

    return st


def prepare_vertical(stream):

    if stream is None or len(stream) == 0:
        raise ValueError("No data in stream")

    vertical = stream.select(component="Z").copy()

    if len(vertical) == 0:
        raise ValueError("No vertical component (Z) found")

    return vertical


def prepare_components(stream):

    if stream is None or len(stream) == 0:
        raise ValueError("No data in stream")

    vertical = stream.select(component="Z").copy()
    north = stream.select(component="N").copy()
    east = stream.select(component="E").copy()

    if len(vertical) == 0:
        raise ValueError("No vertical component (Z) found")

    horizontal = []

    # Match N and E components from the same station.
    for n_trace in north:

        for e_trace in east:

            same_station = (
                n_trace.stats.network == e_trace.stats.network
                and n_trace.stats.station == e_trace.stats.station
                and n_trace.stats.location == e_trace.stats.location
                and n_trace.stats.sampling_rate == e_trace.stats.sampling_rate
                and n_trace.stats.channel[:-1] == e_trace.stats.channel[:-1]
            )

            if not same_station:
                continue

            # Find common time interval.
            start = max(n_trace.stats.starttime,e_trace.stats.starttime,)

            end = min(n_trace.stats.endtime, e_trace.stats.endtime,)

            if end <= start:
                continue

            n = n_trace.slice(start, end)
            e = e_trace.slice(start, end)

            # Verify sample alignment.
            if n.stats.starttime != e.stats.starttime:
                continue

            if n.stats.npts != e.stats.npts:
                continue

            # Calculate horizontal L2 envelope.
            n_data = n.data.astype(np.float64)
            e_data = e.data.astype(np.float64)

            h_data = np.sqrt(n_data ** 2 + e_data ** 2)

            # Create a new horizontal trace.
            h_trace = n.copy()

            h_trace.data = h_data

            h_trace.stats.channel = (n_trace.stats.channel[:-1] + "H")

            horizontal.append(h_trace)

    return {
        "Z": vertical,
        "N": north,
        "E": east,
        "H": horizontal,
    }
