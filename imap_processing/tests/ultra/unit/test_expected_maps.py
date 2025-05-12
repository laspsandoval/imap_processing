import pytest

from imap_processing.ultra.l1b.expected_maps import get_ck_coverage_pairs
from imap_processing import imap_module_directory

@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_get_ck_coverage_pairs():
    """
    Test the get_ck_coverage_pairs function.
    """
    ck_file = (
        imap_module_directory
        / "tests"
        / "spice"
        / "test_data"
        / "sim_1yr_imap_attitude.bc"
    )
    result = get_ck_coverage_pairs(ck_file)
    print("hi")
