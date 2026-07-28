"""Song library backed by a JSON file on disk.

Manages a collection of imported songs. Each song has metadata (title, artist,
model used, etc.) and a directory under ``data/songs/{id}/`` where the original
file and separated stems are stored.
"""

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


_EDITABLE_METADATA_FIELDS = frozenset({"title", "artist", "model_used"})
_GENERATED_SONG_ID_RE = re.compile(r"^[0-9a-f]{12}$")


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
        """Deserialize from a plain dictionary."""
        return cls(**data)


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

        os.makedirs(self._songs_dir, exist_ok=True)
        self._songs_root = os.path.normcase(os.path.realpath(self._songs_dir))

        if os.path.isfile(self._json_path):
            self._load()
        else:
            self._save()

    @property
    def songs(self) -> list[Song]:
        """Return a shallow copy of the song list."""
        return list(self._songs)

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
        song_id = uuid.uuid4().hex[:12]
        song_dir = os.path.join(self._songs_dir, song_id)
        try:
            os.makedirs(song_dir, exist_ok=True)

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
            self._require_safe_song_dir(staged_dir)
            shutil.rmtree(staged_dir)

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
        """Read the JSON index from disk.

        If the file is corrupted, malformed, unreadable, or not valid
        UTF-8, the bad index is preserved as ``library.json.bak`` and the
        library is rebuilt by scanning ``songs/`` so imported stems that
        still exist on disk are recovered instead of silently orphaned.
        """
        try:
            with open(self._json_path, encoding="utf-8") as f:
                data = json.load(f)
            loaded = [Song.from_dict(entry) for entry in data]
            self._songs = [
                song for song in loaded
                if self._is_safe_song_dir(song.stems_path)
                and self._recover_staged_removal(song)
            ]
        except (
            json.JSONDecodeError, TypeError, KeyError,
            OSError, UnicodeDecodeError, ValueError,
        ):
            # Preserve the corrupt index for post-mortem instead of
            # destroying it, then recover what we can from disk.
            try:
                os.replace(self._json_path, self._json_path + ".bak")
            except OSError:
                pass
            self._songs = self._rebuild_from_disk()
            self._save()

    def _rebuild_from_disk(self) -> list["Song"]:
        """Reconstruct library entries by scanning the songs directory.

        Used when the JSON index is lost or corrupt. Each ``songs/<id>/``
        that contains an ``original.*`` file becomes a minimally-populated
        Song; titles fall back to the id and metadata that only lived in
        the index (artist, model) is left blank.
        """
        recovered: list[Song] = []
        try:
            entries = sorted(os.listdir(self._songs_dir))
        except OSError:
            return recovered
        for song_id in entries:
            song_dir = os.path.join(self._songs_dir, song_id)
            if (
                not self._is_safe_song_dir(song_dir)
                or not os.path.isdir(song_dir)
            ):
                continue
            original = None
            for fname in os.listdir(song_dir):
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
                model_used="",
                date_added=datetime.now(timezone.utc).isoformat(),
            ))
        return recovered

    def _save(self) -> None:
        """Write the current song list to the JSON index atomically."""
        tmp_path = self._json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self._songs], f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._json_path)
