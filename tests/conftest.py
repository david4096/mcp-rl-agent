"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configure asyncio for tests
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars():
    """Fixture for setting mock environment variables."""
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["HF_TOKEN"] = "test-token"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)