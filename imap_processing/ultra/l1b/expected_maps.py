import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import spiceypy

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import healpy as hp

from imap_processing.spice.time import sce2met_ns, met_to_sclkticks, sct_to_ttj2000s

SC_ID = -43
TICKS_TO_MS = 1e3 / (5e4)


def plot_all_ultra_counts(input_dir: Path, output_dir: Path):
    """
    Iterate over all Ultra counts CSV files in input_dir,
    plot them with healpy.

    Example:
        input_dir = Path(
    "/Users/lasa6858/Desktop/IMAP_ENA_DirectEvent_Simulations/simulation_outputs-selected/imap_ultra45_366days_logo_test1/")
        output_dir = Path("/Users/lasa6858/Desktop/test/")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(input_dir.glob("*.csv")):
        d = pd.read_csv(file_path)

        energy_str = file_path.stem.split("_E_")[1].split("_counts")[0]
        energy = float(energy_str)

        plt.figure()
        hp.mollview(d["Counts"], title=f"Energy = {energy} keV", cmap='viridis')

        output_file = output_dir / f"{file_path.stem}.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Saved: {output_file}")


def read_all_de_files(de_dir: Path) -> pd.DataFrame:
    """
    Read and combine all DE event files.

    Example:
        de_dir = Path("/Users/lasa6858/Desktop/IMAP_ENA_DirectEvent_Simulations/simulation_outputs-selected/IMAP-Ultra45_r1_L1_V0/")
    """
    de_files = glob.glob(str(de_dir / "*_DEs.txt"))

    sep = r"\s+"
    skiprows = 2
    names = [
        "tdb", "StartX", "PosYSlit", "StopX", "StopY",
        "Energy", "Type", "PH", "SpinPhase", "TOF"
    ]

    dfs = [pd.read_csv(f, sep=sep, skiprows=skiprows, names=names, index_col=False) for f in de_files]

    all_de_df = pd.concat(dfs)
    all_de_df = all_de_df.sort_values("tdb").reset_index(drop=True)
    #epoch = np.asarray(sct_to_ttj2000s(met_to_sclkticks(sce2met_ns(all_de_df["tdb"].values))) * 1e9, dtype=np.int64)
    #all_de_df["epoch"] = epoch

    print(f"Read {len(de_files)} DE files, total rows: {len(all_de_df)}")
    return all_de_df


def get_ck_coverage_pairs(ck_file: Path) -> np.ndarray:
    """
    Get coverage time intervals from a SPICE CK file.
    """
    id_imap_spacecraft = spiceypy.gipool("FRAME_IMAP_SPACECRAFT", 0, 1)
    ck_cover = spiceypy.ckcov(
        str(ck_file), int(id_imap_spacecraft), True, "INTERVAL", 0, "TDB"
    )

    cov_pairs = np.array(ck_cover[:]).reshape((-1, 2))
    num_intervals = spiceypy.wncard(ck_cover)

    return cov_pairs, num_intervals


def generate_repoint_table(ck_file: Path) -> pd.DataFrame:
    """
    Generate repoint table with SCLK seconds, subseconds, and UTC times.
    """
    cov_pairs, _ = get_ck_coverage_pairs(ck_file)
    rp_dict = defaultdict(list)

    for repoint_num in range(cov_pairs.shape[0]):
        start_et = cov_pairs[repoint_num - 1, 1] if repoint_num > 0 else cov_pairs[0, 0]
        end_et = cov_pairs[repoint_num, 0]

        start_met_ns = sce2met_ns(start_et)
        end_met_ns = sce2met_ns(end_et)

        start_sec = int(start_met_ns // 1e9)
        start_subsec = int(start_met_ns % 1e9)

        end_sec = int(end_met_ns // 1e9)
        end_subsec = int(end_met_ns % 1e9)

        start_utc = spiceypy.et2utc(start_et, "ISOC", 6)
        end_utc = spiceypy.et2utc(end_et, "ISOC", 6)

        rp_dict["repoint_start_sec_sclk"].append(start_sec)
        rp_dict["repoint_start_subsec_sclk"].append(start_subsec)
        rp_dict["repoint_end_sec_sclk"].append(end_sec)
        rp_dict["repoint_end_subsec_sclk"].append(end_subsec)
        rp_dict["repoint_start_utc"].append(start_utc)
        rp_dict["repoint_end_utc"].append(end_utc)
        rp_dict["repoint_id"].append(repoint_num)

    rp_df = pd.DataFrame.from_dict(rp_dict)
    return rp_df


def assign_spin_numbers(de_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect spin crossings from spin phase and assign spin numbers to each event.

    Parameters
    ----------
    de_df : pd.DataFrame
        DE event dataframe with 'tdb' and 'SpinPhase' columns.

    Returns
    -------
    de_df : pd.DataFrame
        Original dataframe with new 'spin_number' column.
    """
    spin_phase = de_df["SpinPhase"].values
    tdb_times = de_df["tdb"].values

    # Detect spin start crossings
    phase_diff = np.diff(spin_phase)
    spin_start_indices = np.where(phase_diff < -300)[0] + 1  # spin start at wrap
    spin_start_tdb = tdb_times[spin_start_indices]

    # Assign spin numbers to each event
    spin_numbers = np.searchsorted(spin_start_tdb, tdb_times) - 1
    spin_numbers = np.clip(spin_numbers, 0, None)  # no negatives

    de_df["spin_number"] = spin_numbers.astype(np.uint64)
    return de_df


def assign_pointing_numbers(de_df: pd.DataFrame, ck_file: Path) -> pd.DataFrame:
    """
    Assign pointing numbers to each event based on CK coverage intervals.

    Parameters
    ----------
    de_df : pd.DataFrame
        DE event dataframe with 'tdb' column.
    ck_file : Path
        Path to SPICE CK file.

    Returns
    -------
    de_df : pd.DataFrame
        Dataframe with added 'pointing_number' column.
    """
    cov_pairs, _ = get_ck_coverage_pairs(ck_file)
    pointing_start_et = cov_pairs[:, 0]

    pointing_numbers = np.searchsorted(pointing_start_et, de_df["tdb"].values) - 1
    pointing_numbers = np.clip(pointing_numbers, 0, None)

    de_df["pointing_number"] = pointing_numbers.astype(np.uint64)
    return de_df

def build_full_de_dataframe(de_dir: Path, ck_file: Path) -> pd.DataFrame:
    """
    Read DE files, assign spin and pointing numbers, and return full annotated dataframe.

    Parameters
    ----------
    de_dir : Path
        Directory containing *_DEs.txt files.
    ck_file : Path
        Path to SPICE CK file.

    Returns
    -------
    de_df : pd.DataFrame
        Dataframe with DE events and spin_number, pointing_number, epoch.
    """
    # Step 1: read raw DE events and calculate epoch
    de_df = read_all_de_files(de_dir)

    # Step 2: assign spin numbers
    de_df = assign_spin_numbers(de_df)

    # Step 3: assign pointing numbers
    de_df = assign_pointing_numbers(de_df, ck_file)

    return de_df
