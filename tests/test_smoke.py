"""Smoke tests: the package imports, cheaply."""

from __future__ import annotations

import subprocess
import sys

import gut


def test_version_is_exposed() -> None:
    assert gut.__version__


def test_importing_gut_pulls_in_no_backend_dependencies() -> None:
    """`import gut` must stay instant: no HTTP client, no vendor SDK, no PyTorch until asked."""
    probe = (
        "import sys, gut; "
        "print(sorted(m for m in ('httpx', 'torch', 'transformers', 'typesafe_sdk') "
        "if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]"


def test_every_exported_name_resolves() -> None:
    for name in gut.__all__:
        if name in {"JevBackend", "TransformersBackend", "ZeroShotBackend"}:
            continue  # optional extras; resolved lazily, tested where they are installed
        assert getattr(gut, name) is not None, name


def test_the_version_is_the_same_in_both_places() -> None:
    """The release workflow publishes what pyproject.toml says and refuses a tag that differs."""
    import re
    from pathlib import Path

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert gut.__version__ == match.group(1)


def test_the_mcp_package_is_released_in_lockstep_with_gutfeel() -> None:
    """`gutfeel-mcp` pins the gutfeel it was released with, and gutfeel's extra pins it back."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    main = (root / "pyproject.toml").read_text(encoding="utf-8")
    mcp = (root / "packages" / "gutfeel-mcp" / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"$', main, re.M)
    assert version is not None
    assert f'version = "{version[1]}"' in mcp
    assert f'"gutfeel=={version[1]}"' in mcp
    assert f'mcp = ["gutfeel-mcp=={version[1]}"]' in main


def test_the_mcp_command_is_the_server() -> None:
    import gutfeel_mcp

    from gut import _mcp

    assert gutfeel_mcp.main is _mcp.main


def test_the_registry_entry_names_the_current_release() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    entry = json.loads((root / "server.json").read_text(encoding="utf-8"))
    (package,) = entry["packages"]
    assert entry["version"] == package["version"] == gut.__version__
    assert package["identifier"] == "gutfeel-mcp"
    readme = (root / "packages" / "gutfeel-mcp" / "README.md").read_text(encoding="utf-8")
    assert f"mcp-name: {entry['name']} " in readme or f"mcp-name: {entry['name']} -->" in readme
    assert len(entry["description"]) <= 100
