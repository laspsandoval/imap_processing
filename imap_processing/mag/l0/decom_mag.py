"""Methods for processing raw MAG packets into CDF files for level 0 and level 1a."""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.mag import mag_cdf_attrs
from imap_processing.mag.l0.mag_l0_data import MagL0, Mode

logger = logging.getLogger(__name__)


def decom_packets(packet_file_path: str | Path) -> dict[str, list[MagL0]]:
    """
    Decom MAG data packets using MAG packet definition.

    Parameters
    ----------
    packet_file_path : str
        Path to data packet path with filename.

    Returns
    -------
    data_dict : dict[str, list[MagL0]]
        A dict with 2 keys pointing to lists of MAG L0 data classes. "norm" corresponds
        to  normal mode packets, "burst" corresponds to burst mode packets.
    """
    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/mag/packet_definitions/MAG_SCI_COMBINED.xml"
    )

    packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    mag_parser = parser.PacketParser(packet_definition)

    norm_data = []
    burst_data = []

    with open(packet_file_path, "rb") as binary_data:
        mag_packets = mag_parser.generator(binary_data)

        for packet in mag_packets:
            apid = packet.header["PKT_APID"].derived_value
            if apid in (Mode.BURST, Mode.NORMAL):
                values = [
                    item.derived_value
                    if item.derived_value is not None
                    else item.raw_value
                    for item in packet.data.values()
                ]
                if apid == Mode.NORMAL:
                    norm_data.append(MagL0(CcsdsData(packet.header), *values))
                else:
                    burst_data.append(MagL0(CcsdsData(packet.header), *values))

    return {"norm": norm_data, "burst": burst_data}


def generate_dataset(l0_data: list[MagL0], dataset_attrs: dict) -> xr.Dataset:
    """
    Generate a CDF dataset from the sorted raw L0 MAG data.

    Parameters
    ----------
    l0_data : list[MagL0]
        List of sorted L0 MAG data.

    dataset_attrs : dict
        Global attributes for the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        The xarray dataset with proper CDF attributes and shape.
    """
    # TODO: Correct CDF attributes from email

    vector_data = np.zeros((len(l0_data), len(l0_data[0].VECTORS)))
    shcoarse_data = np.zeros(len(l0_data), dtype="datetime64[ns]")

    support_data = defaultdict(list)

    for index, datapoint in enumerate(l0_data):
        vector_len = len(datapoint.VECTORS)
        if vector_len > vector_data.shape[1]:
            # If the new vector is longer than the existing shape, first reshape
            # vector_data and pad the existing vectors with zeros.
            vector_data = np.pad(
                vector_data,
                (
                    (
                        0,
                        0,
                    ),
                    (0, vector_len - vector_data.shape[1]),
                ),
                "constant",
                constant_values=(0,),
            )
        vector_data[index, :vector_len] = datapoint.VECTORS

        shcoarse_data[index] = met_to_j2000ns(datapoint.SHCOARSE)
        # Add remaining pieces to arrays
        for key, value in dataclasses.asdict(datapoint).items():
            if key not in ("ccsds_header", "VECTORS", "SHCOARSE"):
                support_data[key].append(value)

    # Used in L1A vectors
    direction = xr.DataArray(
        np.arange(vector_data.shape[1]),
        name="direction",
        dims=["direction"],
        attrs=mag_cdf_attrs.raw_direction_attrs.output(),
    )
    # TODO: Epoch here refers to the start of the sample. Confirm that this is
    # what mag is expecting, and if it is, CATDESC needs to be updated.
    epoch_time = xr.DataArray(
        shcoarse_data,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )
    # TODO: raw vectors units
    raw_vectors = xr.DataArray(
        vector_data,
        name="raw_vectors",
        dims=["epoch", "direction"],
        attrs=mag_cdf_attrs.mag_raw_vector_attrs.output(),
    )

    output = xr.Dataset(
        coords={"epoch": epoch_time, "direction": direction},
        attrs=dataset_attrs,
    )

    output["raw_vectors"] = raw_vectors

    for key, value in support_data.items():
        # Time varying values
        if key not in ["SHCOARSE", "VECTORS"]:
            output[key] = xr.DataArray(
                value,
                name=key.lower(),
                dims=["epoch"],
                attrs=dataclasses.replace(
                    mag_cdf_attrs.mag_support_attrs,
                    catdesc=mag_cdf_attrs.catdesc_fieldname_l0[key][0],
                    fieldname=mag_cdf_attrs.catdesc_fieldname_l0[key][1],
                    # TODO: label_axis should be as close to 6 letters as possible
                    label_axis=key,
                    display_type="no_plot",
                ).output(),
            )

    return output
