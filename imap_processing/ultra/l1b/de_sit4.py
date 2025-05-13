"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_de_energy_kev,
    get_de_velocity,
    get_front_y_position,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


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

    de_dict["epoch"] = de_dataset["epoch"].data
    de_dict["spin"] = de_dataset["spin"].data

    species_bin = np.full(len(de_dataset["epoch"]), 1, dtype=np.uint8)

    xf = de_dataset["start_x"].data
    xb = de_dataset["stop_x"].data
    yb = de_dataset["stop_y"].data
    tof = de_dataset["tof"].data

    # Where PosYSlit is True we are "Right/2" and "Left/1" otherwise.
    start_type = np.where(de_dataset["PosYSlit"], 2, 1)

    d, yf = get_front_y_position(
        start_type, yb
    )
    v, vhat, r = get_de_velocity(
        (xf, yf),
        (xb, yb),
        tof,
    )
    de_dict["direct_event_velocity"] = v.astype(np.float32)

    ultra_frame = getattr(SpiceFrame, f"IMAP_ULTRA_45")
    _, sc_dps_velocity, helio_velocity = get_annotated_particle_velocity(
        de_dataset["tdb"].data,
        de_dict["direct_event_velocity"],
        ultra_frame,
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
    )

    de_dict["velocity_dps_sc"] = sc_dps_velocity
    de_dict["velocity_dps_helio"] = helio_velocity
    de_dict["energy_spacecraft"] = get_de_energy_kev(sc_dps_velocity, species_bin)
    de_dict["energy_heliosphere"] = get_de_energy_kev(helio_velocity, species_bin)

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
