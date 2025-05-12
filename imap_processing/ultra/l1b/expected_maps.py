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

from imap_processing.spice.time import sce2met_ns, et_2_datetime, parse_sclk_str

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


def generate_spin_table(ck_file: Path) -> pd.DataFrame:
    """
    Generate spin table with synthetic 15 second spin periods.
    """
    cov_pairs, _ = get_ck_coverage_pairs(ck_file)

    # Full coverage interval
    start_et = cov_pairs[0, 0]
    end_et = cov_pairs[-1, 1]

    # Get first and last SCLK
    _, start_sclk_sec, start_sclk_ticks = parse_sclk_str(spiceypy.sce2s(SC_ID, start_et))
    _, end_sclk_sec, end_sclk_ticks = parse_sclk_str(spiceypy.sce2s(SC_ID, end_et))

    # Generate synthetic spin start times
    spin_start_et = np.arange(start_et, end_et, 15)  # Ideal 15 sec spins
    spin_start_sec = np.arange(start_sclk_sec, end_sclk_sec, 15, dtype=np.uint64)

    spin_dict = dict()
    spin_dict["spin_number"] = np.arange(spin_start_sec.size, dtype=np.uint64)
    spin_dict["spin_start_sec_sclk"] = spin_start_sec
    spin_dict["spin_start_subsec_sclk"] = np.full(
        spin_start_sec.size, start_sclk_ticks * TICKS_TO_MS, dtype=np.uint64
    )
    spin_dict["spin_start_utc"] = np.array(
        [spiceypy.et2utc(et, "ISOC", prec=6).replace("T", " ") for et in spin_start_et]
    )
    spin_dict["spin_period_sec"] = np.full(spin_start_sec.size, 15.0, dtype=np.float64)
    spin_dict["spin_period_valid"] = np.ones(spin_start_sec.size, dtype=np.uint8)
    spin_dict["spin_phase_valid"] = np.ones(spin_start_sec.size, dtype=np.uint8)
    spin_dict["spin_period_source"] = np.zeros(spin_start_sec.size, dtype=np.uint8)
    spin_dict["thruster_firing"] = np.zeros(spin_start_sec.size, dtype=np.uint8)

    # Add thruster firing flags for repointing intervals
    for interval in cov_pairs[1:-1]:  # skip first and last interval
        firing_mask = np.logical_and(
            spin_start_et + 15 >= interval[0],
            spin_start_et < interval[1],
        )
        spin_dict["thruster_firing"][firing_mask] = 1
        spin_dict["spin_period_valid"][firing_mask] = 0

    spin_df = pd.DataFrame.from_dict(spin_dict)

    # Add extra fields for plotting etc.
    spin_df["tdb"] = spin_start_et
    spin_df["datetime"] = et_2_datetime(spin_start_et)
    spin_df = spin_df.set_index("datetime")

    return spin_df
