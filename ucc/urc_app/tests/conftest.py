import sys
from pathlib import Path

# Add URC lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "package" / "lib"))

collect_ignore = ["test_manifests.py"]

import pytest
import responses


@pytest.fixture(autouse=True)
def activate_responses():
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def mock_config():
    return {
        "base_url": "https://mock-api.test",
        "api_key": "test-key-123",
        "username": "testuser",
        "password": "testpass",
    }


@pytest.fixture
def empty_checkpoint():
    return {}
