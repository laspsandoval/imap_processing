"""Tests coverage for imap_processing/spice/time.py"""

import numpy as np
import pytest
import spiceypy as spice

from imap_processing.spice import IMAP_SC_ID
from imap_processing.spice.time import _sct2e_wrapper, met_to_j2000ns


def test_met_to_j2000ns(furnish_time_kernels):
    """Test coverage for met_to_j2000ns function."""
    utc = "2026-01-01T00:00:00.125"
    et = spice.str2et(utc)
    sclk_str = spice.sce2s(IMAP_SC_ID, et)
    seconds, ticks = sclk_str.split("/")[1].split(":")
    # There is some floating point error calculating tick duration from 1 clock
    # tick so average over many clock ticks for better accuracy
    spice_tick_duration = (
        spice.sct2e(IMAP_SC_ID, 1e12) - spice.sct2e(IMAP_SC_ID, 0)
    ) / 1e12
    met = float(seconds) + float(ticks) * spice_tick_duration
    j2000ns = met_to_j2000ns(met)
    assert j2000ns.dtype == np.int64
    np.testing.assert_array_equal(j2000ns, np.array(et * 1e9))


@pytest.mark.parametrize("sclk_ticks", [0.0, np.arange(10)])
def test_sct2e_wrapper(sclk_ticks):
    """Test for `_sct2e_wrapper` function."""
    et = _sct2e_wrapper(sclk_ticks)
    if isinstance(sclk_ticks, float):
        assert isinstance(et, float)
    else:
        assert len(et) == len(sclk_ticks)