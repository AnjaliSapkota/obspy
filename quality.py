
import numpy as np
from scipy.stats import kurtosis


def calculate_quality_metrics(z, env, fs, p_time, s_time, p_sta_trigger, s_sta_trigger):
    if fs <= 0:
        raise ValueError("fs must be positive")

    if len(z.data) == 0:
        raise ValueError("z contains no data")

    if len(env) == 0:
        raise ValueError("env is empty")
  
    if not np.isclose(z.stats.sampling_rate, fs):
        raise ValueError("fs does not match the trace sampling rate")

    if len(z.data) != len(env):
        raise ValueError("z and env must have the same number of samples")

    t0 = z.stats.starttime

    p_idx = int((p_time - t0) * fs)
    s_idx = int((s_time - t0) * fs)

    if not (0 <= p_idx < len(z.data)):
        raise ValueError("p_time is outside the trace time range")

    if not (0 <= s_idx < len(env)):
        raise ValueError("s_time is outside the envelope time range")

    if s_time <= p_time:
        raise ValueError("s_time must be after p_time")

    p_window = int(5.0 * fs)
    s_window = int(5.0 * fs)

    p0 = max(0, p_idx - p_window // 2)
    p1 = min(len(z.data), p_idx + p_window // 2)

    s0 = max(0, s_idx - s_window // 2)
    s1 = min(len(env), s_idx + s_window // 2)

    # Extract waveform windows without unnecessary copies when possible.
    pz = np.asarray(z.data[p0:p1], dtype=float)
    sz = np.asarray(env[s0:s1], dtype=float)

    if len(pz) < 10 or len(sz) < 10:
        raise RuntimeError("Not enough samples for quality analysis")

    if not np.all(np.isfinite(pz)) or not np.all(np.isfinite(sz)):
        raise RuntimeError("Non-finite samples found in quality windows")

    eps = 1e-12

    # Calculate kurtosis for P and S waveform windows.
    p_kurt = float(kurtosis(pz, fisher=True, bias=False))
    s_kurt = float(kurtosis(sz, fisher=True, bias=False))

    def outlier_ratio(data):
        median = np.median(data)
        deviations = np.abs(data - median)
        mad = np.median(deviations)

        if mad <= eps:
            return float(np.mean(deviations > eps))

        zscore = deviations / mad

        return float(np.mean(zscore > 4.45))

    p_outliers = outlier_ratio(pz)
    s_outliers = outlier_ratio(sz)

    sp = float(s_time - p_time)

    checks = []

    if sp < 4.0:
        checks.append("S-P interval is very short")

    if sp > 50.0:
        checks.append("S-P interval exceeds quality limit")

    if p_sta_trigger < 3.5:
        checks.append("Weak P STA/LTA trigger")

    if s_sta_trigger < 2.5:
        checks.append("Weak S STA/LTA trigger")

    if p_outliers > 0.15:
        checks.append("P window contains many outliers")

    if s_outliers > 0.15:
        checks.append("S window contains many outliers")

    # Five quality criteria, each contributing one point.
    score = 0

    if p_sta_trigger >= 3.5:
        score += 1

    if s_sta_trigger >= 2.5:
        score += 1

    if 4.0 <= sp <= 50.0:
        score += 1

    if p_outliers <= 0.15:
        score += 1

    if s_outliers <= 0.15:
        score += 1

    if score >= 4:
        quality = "GOOD"
    elif score >= 3:
        quality = "REVIEW"
    else:
        quality = "REJECT"

    return {
        "quality": quality,
        "quality_score": score,
        "quality_notes": checks,
        "p_kurtosis": p_kurt,
        "s_kurtosis": s_kurt,
        "p_outlier_ratio": p_outliers,
        "s_outlier_ratio": s_outliers,
        "p_sta_trigger": float(p_sta_trigger),
        "s_sta_trigger": float(s_sta_trigger),
    }