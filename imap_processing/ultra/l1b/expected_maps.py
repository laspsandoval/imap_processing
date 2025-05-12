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

    return cov_pairs


def generate_pointing_tables(ck_file: Path):
    """
    Generate pointing table using spice data.
    """
    cov_pairs = get_ck_coverage_pairs(ck_file)
    rp_dict = defaultdict(list)

    for repoint_num in range(cov_pairs.shape[0]):
        rp_dict["repoint_start_time"].append(
            0 if repoint_num == 0 else sce2met_ns(cov_pairs[repoint_num - 1, 1])
        )
        rp_dict["repoint_end_time"].append(
            sce2met_ns(cov_pairs[repoint_num, 0])
            if repoint_num < cov_pairs.shape[0]
            else sce2met_ns(cov_pairs[-1, 0]) + 60 * 60 * 1e9
        )
        rp_dict["repoint_id"].append(repoint_num)

    rp_dict["repoint_start_time"] = (
            np.array(rp_dict["repoint_start_time"], dtype=np.float64) * 1e9
    ).astype(np.uint64)
    rp_dict["repoint_end_time"] = (
            np.array(rp_dict["repoint_end_time"], dtype=np.float64) * 1e9
    ).astype(np.uint64)
    rp_dict["repoint_id"] = np.array(rp_dict["repoint_id"], dtype=np.uint16)

    rp_df = pd.DataFrame.from_dict(rp_dict)
    return rp_df


