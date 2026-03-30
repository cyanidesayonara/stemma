"""User-facing text for import, model download, and separation failures."""


def format_import_error(message: str, max_len: int = 400) -> str:
    """Turn a raw exception or library message into short, readable text."""
    raw = (message or "").strip()
    if not raw:
        return "Something went wrong. Try again."

    low = raw.lower()
    if "disk full" in low or "no space left" in low or "errno 28" in low:
        return "Not enough disk space to finish. Free some space and try again."
    if "permission denied" in low or "errno 13" in low:
        return "Permission denied. Check that the folder is writable."
    if "timed out" in low or "timeout" in low:
        return "The request timed out. Check your network and try again."
    if "network is unreachable" in low or "name or service not known" in low:
        return "Network error. Check your connection and try again."
    if "ssl" in low or "certificate" in low:
        return "Secure connection failed. Check your network or system date."
    if "interruptederror" in low or "cancelled" in low:
        return "The operation was cancelled."
    if "onnxruntimeerror" in low or "runtime_exception" in low:
        oom_phrases = (
            "out of memory",
            "ran out of memory",
            "bad_alloc",
            "bad alloc",
            "std::bad_alloc",
            "failed to allocate",
        )
        if any(p in low for p in oom_phrases):
            return (
                "Not enough memory for stem separation. "
                "Close other apps or try a shorter audio file."
            )
        return (
            "Stem separation failed to initialize. "
            "Try closing other apps to free memory, then retry."
        )
    if (
        "out of memory" in low
        or "ran out of memory" in low
        or "bad_alloc" in low
        or "bad alloc" in low
        or "std::bad_alloc" in low
        or "failed to allocate" in low
    ):
        return (
            "Not enough memory for stem separation. "
            "Close other apps or try a shorter audio file."
        )
    if "http error 404" in low or ("404" in low and "not found" in low):
        return "Download failed: file not found on the server."
    if "http error" in low:
        return "Download failed: server returned an error."

    if len(raw) > max_len:
        return raw[:max_len].rstrip() + "..."
    return raw
