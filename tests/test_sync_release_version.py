"""scripts/sync_release_version.ps1 under both PowerShell editions.

Release CI runs it with pwsh, but the documented local command is often
Windows PowerShell 5.1, where the script used to fail on every write and
still print "Synced", and read the manifest as ANSI.
"""

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_release_version.ps1"

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<Package>
  <Identity
    Name="SanttuNyknen.stemma"
    Version="2.6.0.0"
    ProcessorArchitecture="x64" />
  <Properties>
    <PublisherDisplayName>Santtu Nykänen</PublisherDisplayName>
  </Properties>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" />
  </Dependencies>
</Package>
"""

SHELLS = [
    shell for shell in ("powershell", "pwsh") if shutil.which(shell)
]

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Windows only"),
    pytest.mark.skipif(not SHELLS, reason="no PowerShell"),
]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "version.py").write_text(
        '"""stemma version string."""\n\n__version__ = "2.6.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "msix").mkdir()
    (tmp_path / "msix" / "AppxManifest.xml").write_text(
        MANIFEST, encoding="utf-8",
    )
    return tmp_path


def _run(shell: str, repo: Path, tag: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(repo / "scripts" / SCRIPT.name), "-Tag", tag],
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("shell", SHELLS)
def test_sync_writes_both_files(shell: str, repo: Path) -> None:
    result = _run(shell, repo, "v3.0.0")

    assert result.returncode == 0, result.stderr
    version = (repo / "src" / "version.py").read_bytes()
    assert version == (
        b'"""stemma version string."""\n\n__version__ = "3.0.0"\n'
    )
    manifest = (repo / "msix" / "AppxManifest.xml").read_bytes()
    assert not manifest.startswith(b"\xef\xbb\xbf")
    # The fixture writes CRLF (text mode on Windows), like a checkout
    # with core.autocrlf; both files come out LF-only.
    assert b"\r\n" not in manifest
    text = manifest.decode("utf-8")
    assert 'Version="3.0.0.0"' in text
    assert "Santtu Nykänen" in text
    assert 'MinVersion="10.0.17763.0"' in text


@pytest.mark.parametrize("shell", SHELLS)
def test_sync_rejects_a_bad_tag(shell: str, repo: Path) -> None:
    result = _run(shell, repo, "3.0")

    assert result.returncode != 0
    assert "Synced" not in result.stdout
    assert b'"2.6.0"' in (repo / "src" / "version.py").read_bytes()


@pytest.mark.parametrize("shell", SHELLS)
def test_sync_fails_loudly_when_a_write_fails(shell: str, repo: Path) -> None:
    version = repo / "src" / "version.py"
    version.chmod(0o444)
    try:
        result = _run(shell, repo, "v3.0.0")
    finally:
        version.chmod(0o644)

    assert result.returncode != 0
    assert "Synced" not in result.stdout


@pytest.mark.parametrize("shell", SHELLS)
def test_sync_maps_prerelease_tags_to_distinct_msix_revisions(
    shell: str, repo: Path,
) -> None:
    cases = [
        ("v3.1.0-alpha.1", "3.1.0", "3.1.0.101"),
        ("v3.1.0-beta.1", "3.1.0", "3.1.0.201"),
        ("v3.1.0-beta.2", "3.1.0", "3.1.0.202"),
        ("v3.1.0-rc.1", "3.1.0", "3.1.0.301"),
        ("v3.1.0", "3.1.0", "3.1.0.0"),
    ]
    seen_msix: list[str] = []
    for tag, expected_semver, expected_msix in cases:
        result = _run(shell, repo, tag)
        assert result.returncode == 0, f"{tag}: {result.stderr}"
        version_text = (repo / "src" / "version.py").read_text(encoding="utf-8")
        assert f'__version__ = "{expected_semver}"' in version_text
        manifest_text = (repo / "msix" / "AppxManifest.xml").read_text(
            encoding="utf-8",
        )
        assert f'Version="{expected_msix}"' in manifest_text
        seen_msix.append(expected_msix)

    assert len(set(seen_msix)) == len(cases)
    prerelease_revisions = [int(v.rsplit(".", 1)[1]) for v in seen_msix[:-1]]
    assert prerelease_revisions == sorted(prerelease_revisions)
    assert len(set(prerelease_revisions)) == len(prerelease_revisions)


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    "bad_tag",
    ["v3.1.0-foo", "v3.1.0-beta", "v3.1.0-beta.0", "v3.1.0-beta.100", "v3.1.0+build1"],
)
def test_sync_rejects_unstructured_prerelease_suffix(
    shell: str, repo: Path, bad_tag: str,
) -> None:
    result = _run(shell, repo, bad_tag)

    assert result.returncode != 0
    assert "Synced" not in result.stdout
    assert b'"2.6.0"' in (repo / "src" / "version.py").read_bytes()


def test_manifest_declares_microphone_after_capabilities() -> None:
    """Store builds need microphone device capability after Capability elements."""
    ns_f = "http://schemas.microsoft.com/appx/manifest/foundation/windows10"
    ns_rescap = (
        "http://schemas.microsoft.com/appx/manifest/foundation/"
        "windows10/restrictedcapabilities"
    )
    manifest_path = Path(__file__).resolve().parent.parent / "msix" / "AppxManifest.xml"
    root = ET.parse(manifest_path).getroot()
    capabilities = root.find(f"{{{ns_f}}}Capabilities")
    assert capabilities is not None

    children = list(capabilities)
    assert any(
        el.tag == f"{{{ns_rescap}}}Capability" and el.attrib.get("Name") == "runFullTrust"
        for el in children
    )
    assert any(
        el.tag == f"{{{ns_f}}}DeviceCapability" and el.attrib.get("Name") == "microphone"
        for el in children
    )

    device_tag = f"{{{ns_f}}}DeviceCapability"
    seen_device = False
    for el in children:
        local = el.tag.rsplit("}", 1)[-1]
        if el.tag == device_tag:
            seen_device = True
        elif local.endswith("Capability"):
            assert not seen_device, (
                f"{el.tag} must appear before DeviceCapability per CT_Capabilities"
            )
