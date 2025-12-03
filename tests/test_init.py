"""Test suite for fennomix_novo package."""

import fennomix_novo


def test_version():
    """Test that the version attribute is accessible."""
    assert hasattr(fennomix_novo, "__version__")
    assert isinstance(fennomix_novo.__version__, str)


def test_author():
    """Test that the author attribute is accessible."""
    assert hasattr(fennomix_novo, "__author__")
    assert isinstance(fennomix_novo.__author__, str)
