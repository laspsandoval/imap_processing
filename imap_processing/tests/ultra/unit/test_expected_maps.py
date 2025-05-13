import pytest
import pandas as pd
from pathlib import Path

from imap_processing.ultra.l1b.expected_maps import get_ck_coverage_pairs, generate_repoint_table
from imap_processing.spice.kernels import ensure_spice
from imap_processing.ultra.l1b.expected_maps import build_full_de_dataframe


@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
@ensure_spice
def test_get_ck_coverage_pairs(use_test_metakernel, spice_test_data_path):
    """
    Test the get_ck_coverage_pairs function.
    """
    ck_file = spice_test_data_path / "sim_1yr_imap_attitude.bc"
    result, num_intervals = get_ck_coverage_pairs(ck_file)

    assert len(result) == num_intervals


@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
@ensure_spice
def test_generate_repoint_table(use_test_metakernel, spice_test_data_path):
    """
    Test the get_ck_coverage_pairs function.
    """
    ck_file = spice_test_data_path / "sim_1yr_imap_attitude.bc"
    df = generate_repoint_table(ck_file)


@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
@ensure_spice
def test_build_full_de_dataframe(use_test_metakernel, spice_test_data_path):
    """
    Test the build_full_de_dataframe master function.
    """
    ck_file = spice_test_data_path / "sim_1yr_imap_attitude.bc"
    de_dir = Path("/Users/lasa6858/Desktop/IMAP_ENA_DirectEvent_Simulations/simulation_outputs-selected/IMAP-Ultra45_r1_L1_V0/")

    df = build_full_de_dataframe(de_dir, ck_file)

    # Basic checks
    assert isinstance(df, pd.DataFrame)




