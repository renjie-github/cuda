import pytest
import torch

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def batch_size():
    return 2

@pytest.fixture(scope="session")
def num_beams():
    return 5

@pytest.fixture(scope="session")
def num_beam_groups():
    return 2 