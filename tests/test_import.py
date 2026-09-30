from importlib.metadata import version

import multibubble


def test_package_import():
    assert multibubble.__spec__ is not None


def test_package_installed():
    assert version("multi-bubble-control")
