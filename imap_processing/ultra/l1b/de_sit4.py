"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    StopType,
    determine_species,
    get_coincidence_positions,
    get_ctof,
    get_de_energy_kev,
    get_de_velocity,
    get_efficiency,
    get_energy_pulse_height,
    get_energy_ssd,
    get_eventtimes,
    get_front_x_position,
    get_front_y_position,
    get_fwhm,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_phi_theta,
    get_ssd_back_position_and_tof_offset,
    get_ssd_tof,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

FILLVAL_UINT8 = 255
FILLVAL_FLOAT32 = -1.0e31


def calculate_de(de_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Direct Event Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        L1a dataset containing direct event data.
    name : str
        Name of the l1a dataset.

    Returns
    -------
    dataset : xarray.Dataset
        L1b de dataset.
    """
    de_dict = {}
    sensor = parse_filename_like(name)["sensor"][0:2]

    de_dict["epoch"] = de_dataset["epoch"].data
    de_dict["spin"] = de_dataset["spin"].data

    species_bin = np.full(len(de_dataset["epoch"]), 1, dtype=np.uint8)

    (
        event_times,
        spin_starts,
        spin_period_sec,
    ) = get_eventtimes(
        de_dataset["spin"].data,
        de_dataset["phase_angle"].data,
    )

    xf = de_dataset["start_x"].data
    xb = de_dataset["stop_x"].data
    yb = de_dataset["stop_y"].data
    tof = de_dataset["tof"].data

    d, yf = get_front_y_position(
        de_dataset["start_type"].data, yb
    )
    v, vhat, r = get_de_velocity(
        (xf, yf),
        (xb, yb),
        d,
        tof,
    )
    de_dict["direct_event_velocity"] = v.astype(np.float32)

    ultra_frame = getattr(SpiceFrame, f"IMAP_ULTRA_{sensor}")
    _, sc_dps_velocity, _ = get_annotated_particle_velocity(
        event_times,
        de_dict["direct_event_velocity"],
        ultra_frame,
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
    )

    de_dict["velocity_dps_sc"] = sc_dps_velocity
    de_dict["energy_spacecraft"] = get_de_energy_kev(sc_dps_velocity, species_bin)

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
