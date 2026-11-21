"""Song library backed by a JSON file on disk.

Manages a collection of imported songs. Each song has metadata (title, artist,
model used, etc.) and a directory under ``data/songs/{id}/`` where the original
file and separated stems are stored.
"""

import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, asdict, fields, replace
from datetime import datetime, timezone

from src.separation_state import (
    mark_separation_pending,
    recorded_or_inferred_model,
)


_EDITABLE_METADATA_FIELDS = frozenset({"title", "artist", "model_used"})
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

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Song":
        """Deserialize from a plain dictionary.

        Unknown keys (an index written by a newer version, or edited by
        hand) are ignored and missing text fields default to ``""``, so one
        odd entry cannot discard the whole library. ``id`` and
        ``stems_path`` are required: without them there is no song.
        """
        if not isinstance(data, dict):
            raise TypeError(f"song entry must be an object, got {type(data)}")
        values = {}
        for field in fields(cls):
            value = data.get(field.name)
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise TypeError(f"song field {field.name!r} must be text")
            values[field.name] = value
        if not values["id"] or not values["stems_path"]:
            raise KeyError("song entry needs an id and a stems_path")
        return cls(**values)


class SongLibrary:
    """JSON-backed song library with CRUD operations.

    On construction the data directory structure is created if it does not
    already exist, and any previously persisted songs are loaded into memory.

    Args:
        data_dir: Root data directory (contains ``library.json`` and ``songs/``).
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._songs_dir = os.path.join(data_dir, "songs")
        self._json_path = os.path.join(data_dir, "library.json")
        self._songs: list[Song] = []
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
        staged_dir = None
        if os.path.isdir(song.stems_path):
            staged_dir = os.path.join(
                self._songs_dir,
                f".remove-{song.id}-{uuid.uuid4().hex}",
            )
            self._require_safe_song_dir(staged_dir)
            os.replace(song.stems_path, staged_dir)

        self._songs.pop(song_index)
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

    def update_song(self, song_id: str, **fields: str) -> Song:
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
        self._require_writable()
        previous = {key: getattr(song, key) for key in updates}
        for key, value in updates.items():
            setattr(song, key, value)

        try:
            self._save()
        except Exception:
            for key, value in previous.items():
                setattr(song, key, value)
            raise
        return song

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

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
            self._enter_read_only()
            return

        try:
            data = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(data, list):
                raise ValueError("library index must be a list")
        except ValueError as exc:  # JSON and UTF-8 decode errors included
            logger.warning(
                "Library index %s is damaged: %s", self._json_path, exc,
            )
            self._recover_corrupt_index()
            return

        self._load_entries(data)

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

    def _load_entries(self, data: list) -> None:
        """Turn parsed rows into songs, keeping a backup if any is lost."""
        loaded: list[Song] = []
        lost = False
        rebased = False
        for entry in data:
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
                return
            self._recovery = LibraryRecovery("partial", backup)
            known = {song.id for song in self._songs}
            known.update(entry["id"] for entry in self._held)
            self._songs.extend(
                song for song in self._rebuild_from_disk()
                if song.id not in known
            )
            self._save_quietly()
        elif rebased:
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
        """Write the current song list to the JSON index atomically."""
        self._require_writable()
        tmp_path = self._json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in self._songs] + self._held,
                f, indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._json_path)
