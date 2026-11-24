"""Song library backed by a versioned JSON file on disk.

Manages a collection of imported songs and ordered setlists. Each song has
metadata (title, artist, model used, tags, practice status, etc.) and a
directory under ``data/songs/{id}/`` where the original file and separated
stems are stored.
"""

from collections.abc import Iterable
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone

from src.separation_state import (
    mark_separation_pending,
    recorded_or_inferred_model,
)


LIBRARY_SCHEMA_VERSION = 2
PRACTICE_STATUSES = ("", "learning", "practicing", "ready")
_VALID_PRACTICE_STATUSES = frozenset(PRACTICE_STATUSES)
_SONG_TEXT_FIELDS = (
    "id",
    "title",
    "artist",
    "original_path",
    "stems_path",
    "model_used",
    "date_added",
)
_EDITABLE_METADATA_FIELDS = frozenset({
    "title",
    "artist",
    "model_used",
    "tags",
    "practice_status",
})
_GENERATED_SONG_ID_RE = re.compile(r"^[0-9a-f]{12}$")
# A song id that can name a folder directly below songs/ (no separators,
# drive letters, or leading dot), so a stale path can be rebased onto it.
_REBASABLE_SONG_ID_RE = re.compile(r"^[\w-][\w.-]*$")
_STAGED_REMOVAL_RE = re.compile(r"^\.remove-(.+)-[0-9a-f]{32}$")

# An antivirus scan or a sync client can hold library.json for a moment;
# that is not damage, so the read is retried before giving up.
_READ_ATTEMPTS = 5
_READ_RETRY_DELAY_S = 0.2
_sleep = time.sleep

# Same logger as src.crash_log, which this module cannot import (it pulls
# in Qt through src.data_paths).
logger = logging.getLogger("stemma")


LIBRARY_READ_ONLY_MESSAGE = (
    "The library index could not be read safely when stemma started, so "
    "library changes cannot be saved this session. Close stemma and open "
    "it again."
)


class LibraryReadOnlyError(OSError):
    """The index could not be read safely, so this session cannot save."""


@dataclass(frozen=True)
class LibraryRecovery:
    """What startup had to do because the library index was not healthy.

    ``kind`` is ``"corrupt"`` (unparseable index moved aside and the
    library rebuilt from the song folders), ``"partial"`` (some entries
    were unusable; a copy of the index was kept and the rest loaded), or
    ``"unreadable"`` (the index could not be read or backed up, so it was
    left untouched and nothing is saved this session). ``path`` is the
    backup, or for ``"unreadable"`` the untouched index itself.
    """

    kind: str
    path: str

    @property
    def backup_path(self) -> str:
        """Where the original index can be found."""
        return self.path

    @property
    def message(self) -> str:
        """Readable text for the one-time notice at startup."""
        # File name and folder on separate lines: a whole path is too long
        # for the dialog and wraps after the drive letter.
        name = os.path.basename(self.path)
        folder = os.path.dirname(self.path)
        if self.kind == "corrupt":
            return (
                "stemma could not read its library index, so the library "
                "was rebuilt from the song folders. Your audio and stems "
                "are safe, but song titles and artists may need "
                "re-entering.\n\n"
                f"The damaged index was kept as {name} in:\n{folder}"
            )
        if self.kind == "partial":
            return (
                "Some entries in the library index were damaged. stemma "
                "kept the rest and recovered what it could from the song "
                "folders, so some titles and artists may need "
                "re-entering.\n\n"
                f"A copy of the original index was kept as {name} in:\n"
                f"{folder}"
            )
        if self.kind == "missing":
            return (
                "stemma could not find its library index, so the library "
                "was rebuilt from the song folders. Your audio and stems "
                "are safe, but song titles and artists may need "
                "re-entering.\n\n"
                f"The new index ({name}) is in:\n{folder}"
            )
        if self.kind == "unsaved":
            return (
                "Some entries in the library index were damaged, and stemma "
                "could not keep a backup copy of it. To avoid losing those "
                "entries, library changes are not saved this session. "
                "Close stemma and open it again.\n\n"
                f"The index ({name}) was not changed. It is in:\n{folder}"
            )
        if self.kind == "held":
            return (
                "A song whose removal was interrupted could not be put "
                "back, probably because another program is using its "
                "files. It is kept in the library index and stemma will "
                "try again the next time it starts.\n\n"
                f"Do not delete this folder:\n{name}\n\nIt is in:\n{folder}"
            )
        return (
            "stemma could not read the library index safely. Another "
            "program, such as antivirus or a sync tool, may be using it.\n\n"
            "Songs are listed from their folders for now, with folder "
            "names as titles, and library changes are not saved this "
            "session. Close stemma and open it again to load your titles "
            "and artists.\n\n"
            f"The index ({name}) was not changed. It is in:\n{folder}"
        )


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Strip whitespace, drop blanks, and deduplicate tags case-insensitively."""
    if isinstance(tags, (str, bytes)):
        raise TypeError("tags must be an iterable of strings")
    try:
        iterator = iter(tags)
    except TypeError as exc:
        raise TypeError("tags must be an iterable of strings") from exc
    result: list[str] = []
    seen: set[str] = set()
    for raw in iterator:
        if not isinstance(raw, str):
            raise TypeError(f"tag must be text, got {type(raw)}")
        cleaned = raw.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def normalize_practice_status(status: str | None) -> str:
    """Validate and normalize a song's practice status."""
    if status is None:
        return ""
    if not isinstance(status, str):
        raise TypeError(f"practice_status must be text, got {type(status)}")
    cleaned = status.strip().lower()
    if cleaned not in _VALID_PRACTICE_STATUSES:
        raise ValueError(
            f"Unknown practice status {status!r}; expected one of "
            f"{PRACTICE_STATUSES}"
        )
    return cleaned


def _normalize_single_tag(tag: str) -> str:
    if not isinstance(tag, str):
        raise TypeError(f"tag must be text, got {type(tag)}")
    cleaned = tag.strip()
    if not cleaned:
        raise ValueError("Tag cannot be empty")
    return cleaned


def _dedup_ids(song_ids: Iterable[str]) -> list[str]:
    """Validate and deduplicate song IDs while preserving order."""
    if isinstance(song_ids, (str, bytes)):
        raise TypeError("song_ids must be an iterable of song ID strings")
    try:
        iterator = iter(song_ids)
    except TypeError as exc:
        raise TypeError("song_ids must be an iterable of song ID strings") from exc
    result: list[str] = []
    seen: set[str] = set()
    for sid in iterator:
        if not isinstance(sid, str):
            raise TypeError(f"song id must be text, got {type(sid)}")
        if sid and sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


@dataclass
class Song:
    """Metadata for a single imported song."""

    id: str
    title: str
    artist: str
    original_path: str
    stems_path: str
    model_used: str
    date_added: str
    tags: list[str] = field(default_factory=list)
    practice_status: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Song":
        """Deserialize from a plain dictionary.

        Unknown keys (an index written by a newer version, or edited by
        hand) are ignored, missing text fields default to ``""``, and
        unknown ``practice_status`` values fall back to ``""``, so one
        odd entry cannot discard the whole library. ``id`` and
        ``stems_path`` are required: without them there is no song.
        """
        if not isinstance(data, dict):
            raise TypeError(f"song entry must be an object, got {type(data)}")
        values: dict[str, object] = {}
        for name in _SONG_TEXT_FIELDS:
            value = data.get(name)
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise TypeError(f"song field {name!r} must be text")
            values[name] = value
        if not values["id"] or not values["stems_path"]:
            raise KeyError("song entry needs an id and a stems_path")

        raw_tags = data.get("tags")
        if raw_tags is None:
            values["tags"] = []
        elif not isinstance(raw_tags, list):
            raise TypeError("song field 'tags' must be a list")
        else:
            values["tags"] = normalize_tags(raw_tags)

        raw_status = data.get("practice_status")
        if raw_status is None:
            values["practice_status"] = ""
        elif not isinstance(raw_status, str):
            raise TypeError("song field 'practice_status' must be text")
        else:
            cleaned = raw_status.strip().lower()
            values["practice_status"] = (
                cleaned if cleaned in _VALID_PRACTICE_STATUSES else ""
            )
        return cls(**values)


@dataclass
class Setlist:
    """Ordered collection of song IDs for rehearsal or performance."""

    id: str
    name: str
    song_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)

    def copy(self) -> "Setlist":
        """Return a defensive copy with an independent ``song_ids`` list."""
        return Setlist(id=self.id, name=self.name, song_ids=list(self.song_ids))

    @classmethod
    def from_dict(cls, data: dict) -> "Setlist":
        """Deserialize from a plain dictionary, ignoring unknown keys."""
        if not isinstance(data, dict):
            raise TypeError(f"setlist entry must be an object, got {type(data)}")
        setlist_id = data.get("id")
        name = data.get("name")
        if not isinstance(setlist_id, str) or not isinstance(name, str):
            raise TypeError("setlist 'id' and 'name' must be text")
        clean_id = setlist_id.strip()
        clean_name = name.strip()
        if not clean_id or not clean_name:
            raise ValueError("setlist entry needs non-empty id and name")
        raw_ids = data.get("song_ids")
        if raw_ids is None:
            song_ids: list[str] = []
        elif not isinstance(raw_ids, list):
            raise TypeError("setlist field 'song_ids' must be a list")
        else:
            song_ids = _dedup_ids(raw_ids)
        return cls(id=clean_id, name=clean_name, song_ids=song_ids)


class SongLibrary:
    """JSON-backed song library with CRUD operations.

    On construction the data directory structure is created if it does not
    already exist, and any previously persisted songs and setlists are loaded
    into memory.

    Args:
        data_dir: Root data directory (contains ``library.json`` and ``songs/``).
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._songs_dir = os.path.join(data_dir, "songs")
        self._json_path = os.path.join(data_dir, "library.json")
        self._songs: list[Song] = []
        self._setlists: list[Setlist] = []
        self._writable = True
        self._recovery: LibraryRecovery | None = None
        # Index rows kept on disk but not shown: songs whose interrupted
        # removal could not be restored this launch (see _load_entries).
        self._held: list[dict] = []
        # Every song id the index named when it was read, shown or held.
        self._index_ids: set[str] = set()
        # True only once library.json was read and parsed this launch;
        # staged removal cleanup needs that to know what the index names.
        self._index_read = False

        os.makedirs(self._songs_dir, exist_ok=True)
        self._songs_root = os.path.normcase(os.path.realpath(self._songs_dir))

        if os.path.isfile(self._json_path):
            self._load()
        else:
            # A missing index next to anything in songs/ is lost metadata
            # (for example a damaged index moved aside whose rewrite
            # failed), not a new install: list the folders instead of
            # writing []. Staged removal folders count too, since one may
            # be the only copy of a song.
            self._songs = self._rebuild_from_disk()
            self._setlists = []
            try:
                has_entries = bool(os.listdir(self._songs_dir))
            except OSError:
                has_entries = True
            if has_entries:
                self._recovery = LibraryRecovery("missing", self._json_path)
            self._save()

    @property
    def songs(self) -> list[Song]:
        """Return a shallow copy of the song list."""
        return list(self._songs)

    @property
    def writable(self) -> bool:
        """False when the index could not be read safely this session."""
        return self._writable

    @property
    def recovery(self) -> LibraryRecovery | None:
        """How startup recovered a damaged index, or None if it was fine."""
        return self._recovery

    def _require_writable(self) -> None:
        if not self._writable:
            raise LibraryReadOnlyError(LIBRARY_READ_ONLY_MESSAGE)

    def get_song(self, song_id: str) -> Song | None:
        """Return the song with *song_id*, or ``None`` if not found."""
        for song in self._songs:
            if song.id == song_id:
                return song
        return None

    def _is_safe_song_dir(self, path: str) -> bool:
        """Return whether *path* resolves below this library's songs root."""
        try:
            candidate = os.path.normcase(os.path.realpath(path))
            common = os.path.commonpath([self._songs_root, candidate])
        except (OSError, TypeError, ValueError):
            return False
        return candidate != self._songs_root and common == self._songs_root

    def _require_safe_song_dir(self, path: str) -> None:
        """Reject paths that could make a destructive operation escape."""
        if not self._is_safe_song_dir(path):
            raise ValueError(
                f"Song directory is outside the library songs root: {path}"
            )

    @staticmethod
    def _normalized_path(path: str) -> str:
        """Return a case-normalized absolute path for exact comparisons."""
        return os.path.normcase(os.path.abspath(os.path.normpath(path)))

    def _recover_staged_removal(self, song: Song) -> bool:
        """Restore the newest valid interrupted-removal directory.

        Returns False when valid staged data exists but cannot be restored
        safely, preventing the stale JSON entry from being exposed to startup
        pruning while its recovery data remains staged.
        """
        if not _GENERATED_SONG_ID_RE.fullmatch(song.id):
            return True

        canonical = os.path.join(self._songs_dir, song.id)
        if (
            self._normalized_path(song.stems_path)
            != self._normalized_path(canonical)
            or self._normalized_path(os.path.dirname(song.original_path))
            != self._normalized_path(canonical)
            or os.path.exists(canonical)
        ):
            return True

        try:
            entries = os.listdir(self._songs_dir)
        except OSError:
            return False

        pattern = re.compile(
            rf"^\.remove-{re.escape(song.id)}-([0-9a-f]{{32}})$"
        )
        original_name = os.path.basename(song.original_path)
        candidates: list[tuple[int, str]] = []
        for entry in entries:
            if pattern.fullmatch(entry) is None:
                continue
            candidate = os.path.join(self._songs_dir, entry)
            if (
                not self._is_safe_song_dir(candidate)
                or os.path.islink(candidate)
                or not os.path.isdir(candidate)
                or not os.path.isfile(os.path.join(candidate, original_name))
            ):
                continue
            try:
                modified = os.stat(
                    candidate,
                    follow_symlinks=False,
                ).st_mtime_ns
            except OSError:
                continue
            candidates.append((modified, candidate))

        if not candidates:
            return True

        candidates.sort(reverse=True)
        if (
            len(candidates) > 1
            and candidates[0][0] == candidates[1][0]
        ):
            return False

        try:
            os.replace(candidates[0][1], canonical)
        except OSError:
            return False
        return True

    def add_song(
        self,
        title: str,
        artist: str,
        original_path: str,
        model_used: str = "",
    ) -> Song:
        """Add a new song to the library.

        Creates a per-song directory and persists the updated index.

        Args:
            title: Display title.
            artist: Display artist.
            original_path: Path to the source audio file.
            model_used: Name of the separation model (set later if empty).

        Returns:
            The newly created :class:`Song`.
        """
        self._require_writable()
        song_id = uuid.uuid4().hex[:12]
        song_dir = os.path.join(self._songs_dir, song_id)
        try:
            os.makedirs(song_dir, exist_ok=True)
            # Until a separation writes its completion marker, this song is
            # unfinished work that the startup prune may remove. Written
            # before the copy, so a kill mid-copy is covered too.
            mark_separation_pending(song_dir)

            # Copy the source audio into the song directory so the library is
            # self-contained and does not break if the original file moves.
            ext = os.path.splitext(original_path)[1]
            internal_path = os.path.join(song_dir, f"original{ext}")
            shutil.copy2(original_path, internal_path)
        except OSError:
            if self._is_safe_song_dir(song_dir) and os.path.isdir(song_dir):
                shutil.rmtree(song_dir, ignore_errors=True)
            raise

        song = Song(
            id=song_id,
            title=title,
            artist=artist,
            original_path=internal_path,
            stems_path=song_dir,
            model_used=model_used,
            date_added=datetime.now(timezone.utc).isoformat(),
        )
        self._songs.append(song)
        try:
            self._save()
        except OSError:
            self._songs = [s for s in self._songs if s.id != song_id]
            if self._is_safe_song_dir(song_dir):
                shutil.rmtree(song_dir, ignore_errors=True)
            raise
        return song

    def remove_song(self, song_id: str) -> None:
        """Remove a song by ID, deleting its data directory.

        Raises:
            KeyError: If *song_id* is not in the library.
        """
        song = self.get_song(song_id)
        if song is None:
            raise KeyError(f"Song '{song_id}' not found")

        self._require_writable()
        self._require_safe_song_dir(song.stems_path)
        song_index = self._songs.index(song)
        prev_setlist_songs = [(sl, list(sl.song_ids)) for sl in self._setlists]
        staged_dir = None
        if os.path.isdir(song.stems_path):
            staged_dir = os.path.join(
                self._songs_dir,
                f".remove-{song.id}-{uuid.uuid4().hex}",
            )
            self._require_safe_song_dir(staged_dir)
            os.replace(song.stems_path, staged_dir)

        self._songs.pop(song_index)
        for sl in self._setlists:
            sl.song_ids = [sid for sid in sl.song_ids if sid != song_id]
        try:
            self._save()
        except Exception as save_error:
            if staged_dir is not None:
                try:
                    os.replace(staged_dir, song.stems_path)
                except Exception as restore_error:
                    raise OSError(
                        f"Could not persist removal of song '{song.id}' "
                        f"({save_error}); restoring its directory also "
                        f"failed ({restore_error}). Song data is preserved "
                        f"for recovery at: {staged_dir}"
                    ) from save_error
            self._songs.insert(song_index, song)
            for sl, prev_ids in prev_setlist_songs:
                sl.song_ids = prev_ids
            raise

        if staged_dir is not None:
            # The removal is already saved; a file another program holds
            # must not turn it into an error. The staged folder is left
            # for discard_staged_removals at the next startup.
            try:
                self._require_safe_song_dir(staged_dir)
                shutil.rmtree(staged_dir)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Could not delete removed song folder %s: %s",
                    staged_dir, exc,
                )

    def discard_staged_removals(self) -> None:
        """Delete folders left behind by removals whose delete failed.

        Only ``songs/.remove-<id>-<hex>`` folders whose id the index did not
        name when it was read are touched, and only when the index was
        present and read cleanly: a missing, damaged or unreadable index
        leaves them all alone, since such a folder can be the only copy of
        a song. A held row (a removal that could not be restored) keeps
        its own folder, because the index names its id, but does not
        block cleanup of unrelated ones.
        """
        clean = self._recovery is None or self._recovery.kind == "held"
        if not self._writable or not self._index_read or not clean:
            return
        known = self._index_ids | {song.id for song in self._songs}
        try:
            entries = os.listdir(self._songs_dir)
        except OSError:
            return
        for entry in entries:
            match = _STAGED_REMOVAL_RE.fullmatch(entry)
            if match is None or match.group(1) in known:
                continue
            path = os.path.join(self._songs_dir, entry)
            if (
                not self._is_safe_song_dir(path)
                or os.path.islink(path)
                or not os.path.isdir(path)
            ):
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                logger.warning(
                    "Could not delete removed song folder %s: %s", path, exc,
                )

    def _save_with_rollback(self, restore) -> None:
        """Persist the index, running *restore* before re-raising on failure."""
        try:
            self._save()
        except Exception:
            restore()
            raise

    def _require_songs(self, song_ids: Iterable[str]) -> list[Song]:
        """Resolve *song_ids* in order, raising ``KeyError`` if any is missing."""
        resolved: list[Song] = []
        for sid in _dedup_ids(song_ids):
            song = self.get_song(sid)
            if song is None:
                raise KeyError(f"Song '{sid}' not found")
            resolved.append(song)
        return resolved

    def _require_setlist(self, setlist_id: str) -> Setlist:
        for sl in self._setlists:
            if sl.id == setlist_id:
                return sl
        raise KeyError(f"Setlist '{setlist_id}' not found")

    def update_song(self, song_id: str, **fields) -> Song:
        """Update one or more fields on an existing song.

        Only the supplied keyword arguments are changed; other fields are
        left untouched.  The ``id`` field cannot be changed.

        Raises:
            KeyError: If *song_id* is not in the library.

        Returns:
            The updated :class:`Song`.
        """
        song = self.get_song(song_id)
        if song is None:
            raise KeyError(f"Song '{song_id}' not found")

        updates = {
            key: value
            for key, value in fields.items()
            if key in _EDITABLE_METADATA_FIELDS
        }
        if "tags" in updates:
            updates["tags"] = normalize_tags(updates["tags"])
        if "practice_status" in updates:
            updates["practice_status"] = normalize_practice_status(
                updates["practice_status"]
            )
        self._require_writable()
        previous = {
            key: (
                list(getattr(song, key)) if key == "tags" else getattr(song, key)
            )
            for key in updates
        }
        for key, value in updates.items():
            setattr(song, key, value)

        def restore() -> None:
            for key, value in previous.items():
                setattr(song, key, value)

        self._save_with_rollback(restore)
        return song

    # ------------------------------------------------------------------
    # Tags and practice status
    # ------------------------------------------------------------------

    def all_tags(self) -> list[str]:
        """Return distinct tags across the library sorted case-insensitively."""
        seen: dict[str, str] = {}
        for song in self._songs:
            for tag in song.tags:
                seen.setdefault(tag.casefold(), tag)
        return sorted(seen.values(), key=lambda t: (t.casefold(), t))

    def set_song_tags(self, song_id: str, tags: Iterable[str]) -> Song:
        """Replace the tag list on *song_id*."""
        return self.update_song(song_id, tags=tags)

    def add_tag_to_songs(
        self, song_ids: Iterable[str], tag: str,
    ) -> list[Song]:
        """Add *tag* to every song in *song_ids* in a single save."""
        clean = _normalize_single_tag(tag)
        targets = self._require_songs(song_ids)
        self._require_writable()
        previous = [(song, list(song.tags)) for song in targets]
        key = clean.casefold()
        for song in targets:
            if all(existing.casefold() != key for existing in song.tags):
                song.tags.append(clean)

        def restore() -> None:
            for song, prev_tags in previous:
                song.tags = prev_tags

        self._save_with_rollback(restore)
        return targets

    def remove_tag_from_songs(
        self, song_ids: Iterable[str], tag: str,
    ) -> list[Song]:
        """Remove *tag* (case-insensitively) from every song in *song_ids*."""
        clean = _normalize_single_tag(tag)
        targets = self._require_songs(song_ids)
        self._require_writable()
        previous = [(song, list(song.tags)) for song in targets]
        key = clean.casefold()
        for song in targets:
            song.tags = [
                existing for existing in song.tags if existing.casefold() != key
            ]

        def restore() -> None:
            for song, prev_tags in previous:
                song.tags = prev_tags

        self._save_with_rollback(restore)
        return targets

    def set_practice_status(self, song_id: str, status: str | None) -> Song:
        """Set the practice status badge value on *song_id*."""
        return self.update_song(song_id, practice_status=status)

    # ------------------------------------------------------------------
    # Ordered setlists
    # ------------------------------------------------------------------

    @property
    def setlists(self) -> list[Setlist]:
        """Return defensive copies of all ordered setlists."""
        return [sl.copy() for sl in self._setlists]

    def get_setlist(self, setlist_id: str) -> Setlist | None:
        """Return a defensive copy of *setlist_id*, or ``None``."""
        for sl in self._setlists:
            if sl.id == setlist_id:
                return sl.copy()
        return None

    def create_setlist(
        self, name: str, song_ids: Iterable[str] | None = None,
    ) -> Setlist:
        """Create and persist an ordered setlist."""
        if not isinstance(name, str):
            raise TypeError(f"setlist name must be text, got {type(name)}")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Setlist name cannot be empty")
        ids = _dedup_ids(song_ids) if song_ids is not None else []
        self._require_songs(ids)
        self._require_writable()
        sl = Setlist(id=uuid.uuid4().hex[:12], name=clean_name, song_ids=ids)
        self._setlists.append(sl)
        self._save_with_rollback(self._setlists.pop)
        return sl.copy()

    def rename_setlist(self, setlist_id: str, name: str) -> Setlist:
        """Rename an existing setlist."""
        sl = self._require_setlist(setlist_id)
        if not isinstance(name, str):
            raise TypeError(f"setlist name must be text, got {type(name)}")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Setlist name cannot be empty")
        self._require_writable()
        prev_name = sl.name
        sl.name = clean_name
        self._save_with_rollback(lambda: setattr(sl, "name", prev_name))
        return sl.copy()

    def delete_setlist(self, setlist_id: str) -> None:
        """Delete an existing setlist."""
        sl = self._require_setlist(setlist_id)
        self._require_writable()
        idx = self._setlists.index(sl)
        self._setlists.pop(idx)
        self._save_with_rollback(lambda: self._setlists.insert(idx, sl))

    def add_songs_to_setlist(
        self, setlist_id: str, song_ids: Iterable[str],
    ) -> Setlist:
        """Append *song_ids* (ignoring duplicates already present) to *setlist_id*."""
        sl = self._require_setlist(setlist_id)
        ids = _dedup_ids(song_ids)
        self._require_songs(ids)
        self._require_writable()
        prev_ids = list(sl.song_ids)
        existing = set(sl.song_ids)
        for sid in ids:
            if sid not in existing:
                existing.add(sid)
                sl.song_ids.append(sid)
        self._save_with_rollback(lambda: setattr(sl, "song_ids", prev_ids))
        return sl.copy()

    def remove_song_from_setlist(
        self, setlist_id: str, song_id: str,
    ) -> Setlist:
        """Remove *song_id* from *setlist_id* if present."""
        sl = self._require_setlist(setlist_id)
        self._require_writable()
        prev_ids = list(sl.song_ids)
        sl.song_ids = [sid for sid in sl.song_ids if sid != song_id]
        self._save_with_rollback(lambda: setattr(sl, "song_ids", prev_ids))
        return sl.copy()

    def reorder_setlist(
        self, setlist_id: str, song_ids: Iterable[str],
    ) -> Setlist:
        """Reorder a setlist using an exact permutation of its song IDs."""
        sl = self._require_setlist(setlist_id)
        if isinstance(song_ids, (str, bytes)):
            raise TypeError("song_ids must be an iterable of song ID strings")
        try:
            new_ids = list(song_ids)
        except TypeError as exc:
            raise TypeError("song_ids must be an iterable of song ID strings") from exc
        if any(not isinstance(sid, str) for sid in new_ids):
            raise TypeError("every song id must be text")
        if len(new_ids) != len(set(new_ids)) or set(new_ids) != set(sl.song_ids):
            raise ValueError(
                "song_ids must be an exact permutation of the setlist's songs"
            )
        self._require_writable()
        prev_ids = list(sl.song_ids)
        sl.song_ids = new_ids
        self._save_with_rollback(lambda: setattr(sl, "song_ids", prev_ids))
        return sl.copy()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_index_envelope(data) -> tuple[list, list, bool]:
        """Parse a v1 list or v2 dict index into ``(songs, setlists, migrated)``."""
        if isinstance(data, list):
            return data, [], True
        if isinstance(data, dict):
            version = data.get("schema_version")
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise ValueError(
                    "library index schema_version must be a positive integer"
                )
            raw_songs = data.get("songs")
            if not isinstance(raw_songs, list):
                raise ValueError("library index 'songs' must be a list")
            raw_setlists = data.get("setlists")
            if raw_setlists is None:
                raw_setlists = []
            elif not isinstance(raw_setlists, list):
                raise ValueError("library index 'setlists' must be a list")
            return raw_songs, raw_setlists, False
        raise ValueError("library index must be a list or versioned object")

    def _load(self) -> None:
        """Read the JSON index from disk, recovering what it can.

        Three cases are kept apart, because only one of them is damage:

        - The file cannot be opened (an antivirus scan or a sync client
          holding it). The read is retried; if it still fails, the index is
          left exactly as it is, the songs are listed from their folders
          for this session, and nothing is saved until stemma restarts.
          Rewriting it here would replace good titles with folder names.
        - The file is not valid JSON or UTF-8. It is moved aside under a
          timestamped name, so an earlier backup is never overwritten, and
          the library is rebuilt from ``songs/``.
        - Some entries are unusable. A copy of the index is kept under a
          timestamped name before an index without them is written.

        A song whose stored path points outside this data folder (the
        folder was moved or copied, or migrated from the legacy ``data/``)
        is rebased onto ``songs/<id>`` when that folder exists here.
        """
        try:
            raw = self._read_index_bytes()
        except OSError as exc:
            logger.warning(
                "Library index %s could not be read; leaving it untouched "
                "and not saving this session: %s", self._json_path, exc,
            )
            self._songs = self._rebuild_from_disk()
            self._setlists = []
            self._enter_read_only()
            return

        try:
            data = json.loads(raw.decode("utf-8-sig"))
            raw_songs, raw_setlists, migrated = self._parse_index_envelope(data)
        except ValueError as exc:  # JSON and UTF-8 decode errors included
            logger.warning(
                "Library index %s is damaged: %s", self._json_path, exc,
            )
            self._recover_corrupt_index()
            return

        self._load_entries(raw_songs, raw_setlists, migrated=migrated)

    def _read_index_bytes(self) -> bytes:
        """Read library.json, retrying while another program holds it."""
        for attempt in range(_READ_ATTEMPTS):
            try:
                with open(self._json_path, "rb") as f:
                    return f.read()
            except OSError:
                if attempt == _READ_ATTEMPTS - 1:
                    raise
                _sleep(_READ_RETRY_DELAY_S)
        raise AssertionError("unreachable")

    def _load_entries(
        self,
        raw_songs: list,
        raw_setlists: list = (),
        *,
        migrated: bool = False,
    ) -> None:
        """Turn parsed rows into songs and setlists, backing up if any is lost."""
        loaded: list[Song] = []
        lost = False
        rebased = False
        for entry in raw_songs:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                self._index_ids.add(entry["id"])
            try:
                song = Song.from_dict(entry)
            except (TypeError, KeyError):
                lost = True
                continue
            if not self._is_safe_song_dir(song.stems_path):
                moved = self._rebased(song)
                if moved is None:
                    lost = True
                    continue
                song = moved
                rebased = True
            if self._recover_staged_removal(song):
                loaded.append(song)
            else:
                # Its only copy is a staged removal folder that could not
                # be moved back (a lock, or two equally new candidates).
                # Keep the row in every save so the next launch retries,
                # and keep it out of the list so nothing acts on a folder
                # that is not there.
                logger.warning(
                    "Could not restore song %s from an interrupted "
                    "removal; keeping it for the next launch", song.id,
                )
                self._held.append(song.to_dict())

        parsed_setlists: list[Setlist] = []
        seen_setlist_ids: set[str] = set()
        for raw_sl in raw_setlists:
            try:
                sl = Setlist.from_dict(raw_sl)
            except (TypeError, KeyError, ValueError):
                lost = True
                continue
            if sl.id in seen_setlist_ids:
                lost = True
                continue
            seen_setlist_ids.add(sl.id)
            parsed_setlists.append(sl)

        self._songs = loaded
        self._index_read = True
        if self._held:
            # Replaced below if rows were also lost: that notice matters
            # more, and held rows are kept either way.
            self._recovery = LibraryRecovery(
                "held", self._staged_folder_of(self._held[0]["id"]),
            )

        if lost:
            backup = self._backup_path("partial")
            try:
                shutil.copyfile(self._json_path, backup)
            except OSError as exc:
                # Without a copy, writing an index that leaves rows out
                # would lose them for good.
                logger.warning(
                    "Could not back up library index to %s: %s", backup, exc,
                )
                self._writable = False
                self._recovery = LibraryRecovery("unsaved", self._json_path)
                known = {song.id for song in self._songs}
                known.update(entry["id"] for entry in self._held)
                for sl in parsed_setlists:
                    sl.song_ids = [sid for sid in sl.song_ids if sid in known]
                self._setlists = parsed_setlists
                return
            self._recovery = LibraryRecovery("partial", backup)
            known = {song.id for song in self._songs}
            known.update(entry["id"] for entry in self._held)
            self._songs.extend(
                song for song in self._rebuild_from_disk()
                if song.id not in known
            )

        known = {song.id for song in self._songs}
        known.update(entry["id"] for entry in self._held)
        setlists_cleaned = False
        for sl in parsed_setlists:
            filtered = [sid for sid in sl.song_ids if sid in known]
            if len(filtered) != len(sl.song_ids):
                sl.song_ids = filtered
                setlists_cleaned = True
        self._setlists = parsed_setlists

        if lost or rebased or migrated or setlists_cleaned:
            self._save_quietly()

    def _staged_folder_of(self, song_id: str) -> str:
        """Return the newest-named staged removal folder for *song_id*."""
        prefix = f".remove-{song_id}-"
        try:
            names = sorted(
                name for name in os.listdir(self._songs_dir)
                if name.startswith(prefix)
            )
        except OSError:
            names = []
        if not names:
            return os.path.join(self._songs_dir, prefix + "*")
        return os.path.join(self._songs_dir, names[-1])

    def _rebased(self, song: Song) -> Song | None:
        """Return *song* with its paths moved under this songs root.

        Used when the stored path is outside the root, as after the data
        folder was moved or copied. Returns None when ``songs/<id>`` does
        not exist here or the id could not name a folder below songs/.
        """
        if not _REBASABLE_SONG_ID_RE.fullmatch(song.id):
            return None
        new_dir = os.path.join(self._songs_dir, song.id)
        if not self._is_safe_song_dir(new_dir) or not os.path.isdir(new_dir):
            return None
        original = song.original_path
        if original:
            candidate = os.path.join(new_dir, os.path.basename(original))
            was_inside = (
                self._normalized_path(os.path.dirname(original))
                == self._normalized_path(song.stems_path)
            )
            if was_inside or os.path.isfile(candidate):
                original = candidate
        logger.info(
            "Rebased song %s from %s to %s",
            song.id, song.stems_path, new_dir,
        )
        return replace(song, stems_path=new_dir, original_path=original)

    def _recover_corrupt_index(self) -> None:
        """Move an unparseable index aside, then rebuild from the folders."""
        self._setlists = []
        backup = self._backup_path("corrupt")
        try:
            os.replace(self._json_path, backup)
        except OSError:
            try:
                shutil.copyfile(self._json_path, backup)
            except OSError as exc:
                logger.warning(
                    "Could not back up damaged library index to %s: %s",
                    backup, exc,
                )
                self._songs = self._rebuild_from_disk()
                self._enter_read_only()
                return
        self._recovery = LibraryRecovery("corrupt", backup)
        self._songs = self._rebuild_from_disk()
        self._save_quietly()

    def _enter_read_only(self) -> None:
        self._writable = False
        self._recovery = LibraryRecovery("unreadable", self._json_path)

    def _backup_path(self, tag: str) -> str:
        """Return a backup name next to the index that does not exist yet."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{self._json_path}.{tag}-{stamp}"
        path = base + ".bak"
        counter = 2
        while os.path.lexists(path):
            path = f"{base}-{counter}.bak"
            counter += 1
        return path

    def _save_quietly(self) -> None:
        """Save during load; a failed write is retried by the next change."""
        try:
            self._save()
        except OSError as exc:
            logger.warning(
                "Could not save library index %s: %s", self._json_path, exc,
            )

    def _rebuild_from_disk(self) -> list["Song"]:
        """Reconstruct library entries by scanning the songs directory.

        Used when the JSON index is lost or corrupt. Each ``songs/<id>/``
        that contains an ``original.*`` file becomes a minimally-populated
        Song; titles fall back to the id, the artist is left blank, and
        the model is read from the completion marker or inferred from the
        stems on disk.
        """
        recovered: list[Song] = []
        try:
            entries = sorted(os.listdir(self._songs_dir))
        except OSError:
            return recovered
        for song_id in entries:
            if song_id.startswith("."):
                # Staged removals (.remove-<id>-...) and other internals,
                # not songs.
                continue
            song_dir = os.path.join(self._songs_dir, song_id)
            if (
                not self._is_safe_song_dir(song_dir)
                or not os.path.isdir(song_dir)
            ):
                continue
            original = None
            try:
                names = os.listdir(song_dir)
            except OSError:
                continue
            for fname in names:
                if fname.startswith("original."):
                    original = os.path.join(song_dir, fname)
                    break
            if original is None:
                continue
            recovered.append(Song(
                id=song_id,
                title=song_id,
                artist="",
                original_path=original,
                stems_path=song_dir,
                # Without it, startup pruning cannot tell a finished song
                # from an interrupted one (#181).
                model_used=recorded_or_inferred_model(song_dir),
                date_added=datetime.now(timezone.utc).isoformat(),
            ))
        return recovered

    def _save(self) -> None:
        """Write the current library state to the versioned JSON index."""
        self._require_writable()
        payload = {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "songs": [s.to_dict() for s in self._songs] + self._held,
            "setlists": [sl.to_dict() for sl in self._setlists],
        }
        tmp_path = self._json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._json_path)
