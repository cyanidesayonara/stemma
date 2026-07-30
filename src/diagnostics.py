"""Source and frozen-build diagnostics for release smoke checks."""

import os
import sys
import tempfile

from src.version import __version__


def collect_diagnostics() -> dict[str, str | list[str]]:
    """Return app and ONNX Runtime version/provider diagnostics."""
    # Deferred import: diagnostics is an explicit smoke path, while normal
    # startup must not eagerly load the heavy ONNX Runtime package.
    import onnxruntime as ort

    return {
        "stemma_version": __version__,
        "onnxruntime_version": ort.__version__,
        "available_providers": list(ort.get_available_providers()),
    }


def format_diagnostics(
    diagnostics: dict[str, str | list[str]],
) -> str:
    """Format release diagnostics as stable, human-readable text."""
    providers = ", ".join(diagnostics["available_providers"])
    return "\n".join([
        f"stemma version: {diagnostics['stemma_version']}",
        f"ONNX Runtime version: {diagnostics['onnxruntime_version']}",
        f"Available ONNX providers: {providers}",
    ])


def diagnostics_requested(argv: list[str]) -> bool:
    """Return whether this process should run release diagnostics."""
    return any(
        arg == "--diagnostics"
        or arg == "--diagnostics-file"
        or arg.startswith("--diagnostics-file=")
        for arg in argv[1:]
    )


def _diagnostics_file_path(argv: list[str]) -> str | None:
    """Parse the optional diagnostics output path or raise ``ValueError``."""
    for index, arg in enumerate(argv[1:], start=1):
        if arg.startswith("--diagnostics-file="):
            path = arg.partition("=")[2]
            if not path:
                raise ValueError("--diagnostics-file requires a path")
            return path
        if arg == "--diagnostics-file":
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError("--diagnostics-file requires a path")
            return argv[index + 1]
    return None


def _write_diagnostics_file(path: str, content: str) -> None:
    """Atomically write UTF-8 diagnostics to *path*."""
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(absolute)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, absolute)
    except Exception:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    """Write requested diagnostics and return a process exit code."""
    argv = sys.argv if argv is None else argv
    try:
        output_path = _diagnostics_file_path(argv)
    except ValueError as exc:
        if sys.stderr is not None:
            print(str(exc), file=sys.stderr, flush=True)
        return 2

    console_requested = "--diagnostics" in argv[1:]
    if not console_requested and output_path is None:
        return 2

    content = format_diagnostics(collect_diagnostics())
    if output_path is not None:
        _write_diagnostics_file(output_path, content)
    if console_requested and sys.stdout is not None:
        print(content, file=sys.stdout, flush=True)
    return 0
