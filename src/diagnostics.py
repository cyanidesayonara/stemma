"""Source and frozen-build diagnostics for release smoke checks."""

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
    return "--diagnostics" in argv[1:]


def main() -> int:
    """Print diagnostics and return a process exit code."""
    print(format_diagnostics(collect_diagnostics()), flush=True)
    return 0
