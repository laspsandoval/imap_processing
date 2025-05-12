import pytest

from imap_processing.ultra.l1b.expected_maps import get_ck_coverage_pairs, generate_pointing_tables
from imap_processing.spice.kernels import ensure_spice


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
def test_generate_pointing_tables(use_test_metakernel, spice_test_data_path):
    """
    Test the get_ck_coverage_pairs function.
    """
    ck_file = spice_test_data_path / "sim_1yr_imap_attitude.bc"
    rp_df = generate_pointing_tables(ck_file)

    print('hi')
