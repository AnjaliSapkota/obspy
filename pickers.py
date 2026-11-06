import numpy as np
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

def aic_pick(trace_data, fs, approximate_idx, search_before=1.0, search_after=0.5, max_shift_sec=0.5,):
    trace_data = np.asarray(trace_data, dtype=float)
    n = len(trace_data)
    if n == 0:
        raise ValueError("trace_data is empty")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not (0 <= approximate_idx < n):
        raise ValueError("approximate_idx out of bounds")
    i0 = max(1, approximate_idx - int(search_before * fs))
    i1 = min(n - 2, approximate_idx + int(search_after * fs))
    if i1 <= i0 + 10:
        return approximate_idx

    x = trace_data[i0:i1]

    if not np.all(np.isfinite(x)):
        return approximate_idx

    length = len(x)

    # Center the data to improve numerical stability.
    x = x - np.mean(x)

    # Cumulative sums allow variance calculations at every split point.
    cumulative_sum = np.cumsum(x, dtype=np.float64)
    cumulative_sq_sum = np.cumsum(x * x, dtype=np.float64)

    k = np.arange(1, length - 1)

    sum_left = cumulative_sum[k - 1]
    sum_right = cumulative_sum[-1] - sum_left

    sq_left = cumulative_sq_sum[k - 1]
    sq_right = cumulative_sq_sum[-1] - sq_left

    var_left = sq_left / k - (sum_left / k) ** 2

    right_length = length - k

    var_right = (sq_right / right_length - (sum_right / right_length) ** 2)

    valid = (var_left > 0) & (var_right > 0)

    if not np.any(valid):
        return approximate_idx

    aic = np.full(len(k), np.inf)
    aic[valid] = (k[valid] * np.log10(var_left[valid]) + right_length[valid] * np.log10(var_right[valid]))
    local_idx = int(np.argmin(aic))
    edge = max(2, int(0.05 * length))
    split_idx = int(k[local_idx])
    if split_idx < edge or split_idx >= length - edge:
        return approximate_idx
    pick_idx = i0 + int(k[local_idx])

    if abs(pick_idx - approximate_idx) > max_shift_sec * fs:
        return approximate_idx

    return pick_idx


def pick_p(z, fs, start, sta=1.0, lta=10.0, thr_on=3.5, thr_off=1.5, aic_before=1.0, aic_after=0.5,):
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not np.isclose(z.stats.sampling_rate, fs):
        raise ValueError("fs does not match the trace sampling rate")
    if len(z.data) == 0:
        raise ValueError("z contains no data")
    start_idx = int((start - z.stats.starttime) * fs)
    if not (0 <= start_idx < len(z.data)):
        raise ValueError("start is outside the trace time range")

    nsta = max(1, int(sta * fs))
    nlta = max(nsta + 1, int(lta * fs))
    cft = recursive_sta_lta(z.data, nsta, nlta)
    p_idx = None
    for on, _off in trigger_onset(cft, thr_on, thr_off):
        t = z.stats.starttime + on / fs

        if t >= start:
            p_idx = int(on)
            break

    if p_idx is None:
        return None, cft, None, None

    p_idx_aic = aic_pick(z.data, fs, p_idx, search_before=aic_before, search_after=aic_after, max_shift_sec=0.5,)

    p_time_sta = z.stats.starttime + p_idx / fs
    p_time_aic = z.stats.starttime + p_idx_aic / fs

    trigger_value = float(cft[p_idx])

    return p_time_aic, cft, p_time_sta, trigger_value


def pick_s(env, fs, t0, p_time, sta=1.0, lta=8.0, min_sp_sec=1.0, max_sp_sec=60.0, thr_on=2.5, thr_off=1.0, aic_before=1.5, aic_after=4.0,):
    if fs <= 0:
        raise ValueError("fs must be positive")
    if len(env) == 0:
        raise ValueError("env is empty")
    p_idx = int((p_time - t0) * fs)
    if not (0 <= p_idx < len(env)):
        raise ValueError("p_time is outside the envelope time range")

    nsta = max(1, int(sta * fs))
    nlta = max(nsta + 1, int(lta * fs))
    cft = recursive_sta_lta(env, nsta, nlta)
    i0 = p_idx + int(min_sp_sec * fs)
    i1 = min(p_idx + int(max_sp_sec * fs), len(cft) - 1)
    if i0 >= i1 or i0 >= len(cft):
        return None, cft, None, None
    s_idx = None
    for on, off in trigger_onset(cft, thr_on, thr_off):
        if on > i1:
            break

        if on < i0:
            continue

        s_idx = int(on)
        break

    if s_idx is None:
        return None, cft, None, None

    s_idx_aic = aic_pick(env, fs, s_idx, search_before=aic_before, search_after=aic_after)

    s_time_sta = t0 + s_idx / fs
    s_time_aic = t0 + s_idx_aic / fs

    trigger_value = float(cft[s_idx])

    return s_time_aic, cft, s_time_sta, trigger_value