from __future__ import annotations

import numpy as np

# A standard log-distance path-loss model. Values are deliberately explicit so the
# simulator can report its assumptions and be replaced by real calibration later.
RSSI_AT_ONE_M_DBM = -59.0
PATH_LOSS_EXPONENT = 2.15


def range_to_rssi(distance_m: np.ndarray, calibration_bias: float = 0.0) -> np.ndarray:
    """Convert a positive range in metres to an idealized BLE RSSI observation."""
    safe_distance = np.maximum(np.asarray(distance_m, dtype=float), 0.25)
    return RSSI_AT_ONE_M_DBM - 10.0 * PATH_LOSS_EXPONENT * np.log10(safe_distance) + calibration_bias


def rssi_to_range(rssi_dbm: np.ndarray) -> np.ndarray:
    """The f(r_i) range conversion used by the localization objective."""
    rssi = np.asarray(rssi_dbm, dtype=float)
    return 10.0 ** ((RSSI_AT_ONE_M_DBM - rssi) / (10.0 * PATH_LOSS_EXPONENT))
