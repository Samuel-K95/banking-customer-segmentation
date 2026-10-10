from customer_segmentation import __version__


def test_version_semver_like():
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) >= 2
