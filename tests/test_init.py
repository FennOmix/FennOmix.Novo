"""Test suite for foxnovo package."""

import foxnovo


def test_version():
    """Test that the version attribute is accessible."""
    assert hasattr(foxnovo, "__version__")
    assert isinstance(foxnovo.__version__, str)


def test_author():
    """Test that the author attribute is accessible."""
    assert hasattr(foxnovo, "__author__")
    assert isinstance(foxnovo.__author__, str)
