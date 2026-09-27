"""scripts/sync_release_version.ps1 under both PowerShell editions.

Release CI runs it with pwsh, but the documented local command is often
Windows PowerShell 5.1, where the script used to fail on every write and
still print "Synced", and read the manifest as ANSI.
"""

import shutil
import subprocess
import sys
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
