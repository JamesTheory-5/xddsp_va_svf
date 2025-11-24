# xddsp_va_svf
```python
# xddsp_vasvf.py
# ============================================================
# VA-SVF (Virtual Analog State Variable Filter)
# XDDSP-core style: pure functional, tuple-state, Numba JIT
# ============================================================

import math
import numpy as np
from numba import njit


# ============================================================
# MODULE DESCRIPTION
# ============================================================
"""
VA-SVF (Virtual Analog State Variable Filter)

What the module does:
- Implements a zero-delay feedback / TPT state-variable filter
- Supports multiple responses: LP, BP, HP, Notch, Allpass, Peak, Shelf
- Uses tuple-only state and parameter layout, pure functional tick/process

Fully functional, differentiable (outside JIT), stateless except for explicit
tuple state (s1, s2).
"""


# ============================================================
# STATE LAYOUT (TUPLE ONLY)
# ============================================================
# state = (
#     s1,   # first integrator state
#     s2,   # second integrator state
# )
#
# params = (
#     fs,   # sample rate (float)
#     g,    # TPT analog frequency transform coefficient
#     g1,   # 2*R + g
#     R,    # 1 / (2 * Q)
#     M,    # linear gain factor for shelf/peak (10 ** (gain_dB / 20))
#     d,    # 1 / (1 + 2 R g + g^2)
#     mode, # integer mode selector (0..6)
# )
#
# All shapes computed outside JIT.
# Only scalars in state/params inside JIT.


# ============================================================
# HELPER: coeff computation (outside JIT)
# ============================================================
def _vasvf_coeffs(fs, f0, Q, gain_dB):
    """
    Compute VA-SVF scalar coefficients (Python side).

    Clamps:
    - f0 in [1e-6, 0.49 * fs]
    - Q  >= 1e-6
    """
    fs = float(fs)
    f0 = float(f0)
    Q = float(Q)
    gain_dB = float(gain_dB)

    # clamps
    if f0 < 1e-6:
        f0 = 1e-6
    nyq = 0.5 * fs
    if f0 > 0.98 * nyq:  # keep a little margin
        f0 = 0.98 * nyq

    if Q < 1e-6:
        Q = 1e-6

    R = 1.0 / (2.0 * Q)
    M = 10.0 ** (gain_dB / 20.0)
    g = math.tan(math.pi * f0 / fs)
    g1 = 2.0 * R + g
    d = 1.0 / (1.0 + 2.0 * R * g + g * g)
    return fs, g, g1, R, M, d


# ============================================================
# INIT
# ============================================================
def vasvf_init(fs, f0, Q=0.707, gain_dB=0.0, mode=0):
    """
    Returns (state, params).

    Parameters
    ----------
    fs : float
        Sample rate
    f0 : float
        Cutoff / center frequency (Hz)
    Q : float
        Quality factor
    gain_dB : float
        Gain in dB for peak/shelf modes
    mode : int
        0 = LP, 1 = BP, 2 = HP,
        3 = Notch, 4 = Allpass, 5 = Peak, 6 = Shelf
    """
    fs, g, g1, R, M, d = _vasvf_coeffs(fs, f0, Q, gain_dB)

    # initial integrator state
    state = (
        0.0,  # s1
        0.0,  # s2
    )

    # params tuple
    params = (
        fs,
        g,
        g1,
        R,
        M,
        d,
        int(mode),
    )

    return state, params


# ============================================================
# UPDATE PARAMS / UPDATE STATE (functionally)
# ============================================================
def vasvf_update_state(
    state,
    params,
    fs=None,
    f0=None,
    Q=None,
    gain_dB=None,
    mode=None,
    reset=False,
):
    """
    Functional parameter/state update.
    No mutation allowed.
    Returns (new_state, new_params).

    Any argument left as None keeps its previous value.
    If reset=True, integrator state is cleared to zero.
    """

    old_fs, old_g, old_g1, old_R, old_M, old_d, old_mode = params

    # derive effective scalar values
    fs_eff = old_fs if fs is None else float(fs)
    # if no new f0/Q/gain provided, reuse old coeffs *as is*
    # otherwise recompute full set from fresh (f0, Q, gain_dB)
    if f0 is None and Q is None and gain_dB is None:
        g_eff = old_g
        g1_eff = old_g1
        R_eff = old_R
        M_eff = old_M
        d_eff = old_d
    else:
        # we don't track f0/Q/gain separately, so caller must
        # pass all three or accept reuse of old ones if None.
        # To keep it simple, if any is None, we infer it from
        # the previous coeff set via a best-effort guess; or
        # we just require all to be non-None for "nice" use.
        # Here we choose the simpler: require all non-None.
        if (f0 is None) or (Q is None) or (gain_dB is None):
            raise ValueError(
                "vasvf_update_state: if you update frequency, Q, or gain, "
                "you must provide all three (f0, Q, gain_dB)."
            )
        fs_, g_eff, g1_eff, R_eff, M_eff, d_eff = _vasvf_coeffs(
            fs_eff, f0, Q, gain_dB
        )
        fs_eff = fs_

    mode_eff = old_mode if mode is None else int(mode)

    # new state
    if reset:
        new_state = (0.0, 0.0)
    else:
        new_state = state

    new_params = (
        fs_eff,
        g_eff,
        g1_eff,
        R_eff,
        M_eff,
        d_eff,
        mode_eff,
    )

    return new_state, new_params


# ============================================================
# TICK (NUMBA)
# ============================================================
@njit(cache=True, fastmath=True)
def vasvf_tick(x, state, params):
    """
    Pure functional VA-SVF tick.

    Inputs
    ------
    x : float
        current input sample
    state : tuple
        (s1, s2) current SVF integrator states
    params : tuple
        (fs, g, g1, R, M, d, mode) constant for this block

    Returns
    -------
    y : float
        selected mode output sample
    new_state : tuple
        (s1, s2) next-state
    """
    # unpack state
    s1 = state[0]
    s2 = state[1]

    # unpack params
    # fs not strictly needed in tick, but kept for completeness
    fs = params[0]
    g = params[1]
    g1 = params[2]
    R = params[3]
    M = params[4]
    d = params[5]
    mode = params[6]

    # --- core VA-SVF equations (ZDF / TPT) ---
    hp = (x - g1 * s1 - s2) * d
    v1 = g * hp
    bp = v1 + s1
    s1_new = bp + v1
    v2 = g * bp
    lp = v2 + s2
    s2_new = lp + v2

    # --- derived responses ---
    bp1 = 2.0 * R * bp
    notch = x - bp1
    allpass = x - 2.0 * bp1  # == x - 4 R bp
    peak = lp - hp
    k = (M ** -2.0) - 1.0
    shelf = x + k * bp1

    # --- mode selection WITHOUT Python branching ---

    # Each flag is 1.0 when mode == N, else 0.0
    # Numba happily treats (mode == N) as boolean and casts to float.
    f_lp = 1.0 * (mode == 0)
    f_bp = 1.0 * (mode == 1)
    f_hp = 1.0 * (mode == 2)
    f_notch = 1.0 * (mode == 3)
    f_ap = 1.0 * (mode == 4)
    f_peak = 1.0 * (mode == 5)
    f_shelf = 1.0 * (mode == 6)

    # weighted sum of all responses
    y = (
        lp * f_lp
        + bp * f_bp
        + hp * f_hp
        + notch * f_notch
        + allpass * f_ap
        + peak * f_peak
        + shelf * f_shelf
    )

    new_state = (s1_new, s2_new)
    return y, new_state


# ============================================================
# PROCESS (NUMBA) — lax.scan wrapper
# ============================================================
@njit(cache=True, fastmath=True)
def vasvf_process(x, state, params):
    """
    Vectorized per-block processing.
    Equivalent to lax.scan over vasvf_tick().

    Parameters
    ----------
    x : ndarray shape (N,)
        input signal
    state : tuple
        (s1, s2)
    params : tuple
        (fs, g, g1, R, M, d, mode)
    """
    N = x.shape[0]
    y = np.empty(N, dtype=np.float64)

    s = state
    for i in range(N):
        yi, s = vasvf_tick(x[i], s, params)
        y[i] = yi

    return y, s


# ============================================================
# SMOKE TEST
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000.0
    f0 = 1000.0
    Q = 0.707
    gain_dB = 0.0
    mode = 0  # lowpass

    # construct module
    state, params = vasvf_init(fs, f0, Q, gain_dB, mode)

    # test input: impulse
    N = 48000
    x = np.zeros(N, dtype=np.float64)
    x[0] = 1.0

    # run
    y, state = vasvf_process(x, state, params)

    # plot
    t = np.arange(N) / fs
    plt.plot(t, y)
    plt.title("VA-SVF Impulse Response (mode=%d)" % mode)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    # audio test (optional)
    try:
        import sounddevice as sd

        sd.play(y, int(fs))
        sd.wait()
    except Exception as e:
        print("Audio playback unavailable:", e)

```
