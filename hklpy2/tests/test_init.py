"""Test the package constructor."""

from unittest.mock import Mock

import pytest
import setuptools_scm

import hklpy2


class TestException: ...


@pytest.mark.parametrize(
    "version, exception",
    [
        ["111.111.111", None],
        # TODO: How to trigger the Exception handling in hklpy2._get_version()?  See GH prjemian/Chewacla repo.
        ["222.222.222", LookupError],
        ["333.333.333", TestException],
    ],
)
def test_get_version(version, exception):
    hklpy2._get_version = Mock(return_value=version)
    if exception is not None:
        setuptools_scm.get_version = exception
    assert hklpy2._get_version() == version
