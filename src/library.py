"""Song library backed by a JSON file on disk.

Manages a collection of imported songs. Each song has metadata (title, artist,
model used, etc.) and a directory under ``data/songs/{id}/`` where the original
file and separated stems are stored.
"""

import json
import os
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


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
        os.makedirs(song_dir, exist_ok=True)

        # Copy the source audio into the song directory so the library is
        # self-contained and does not break if the original file moves.
        ext = os.path.splitext(original_path)[1]
        internal_path = os.path.join(song_dir, f"original{ext}")
        shutil.copy2(original_path, internal_path)

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
        self._save()
        return song

    def remove_song(self, song_id: str) -> None:
        """Remove a song by ID, deleting its data directory.

        Raises:
            KeyError: If *song_id* is not in the library.
        """
        song = self.get_song(song_id)
        if song is None:
            raise KeyError(f"Song '{song_id}' not found")

        if os.path.isdir(song.stems_path):
            shutil.rmtree(song.stems_path)

        self._songs = [s for s in self._songs if s.id != song_id]
        self._save()

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

        fields.pop("id", None)
        for key, value in fields.items():
            if hasattr(song, key):
                setattr(song, key, value)

        self._save()
        return song

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the JSON index from disk."""
        with open(self._json_path, encoding="utf-8") as f:
            data = json.load(f)
        self._songs = [Song.from_dict(entry) for entry in data]

    def _save(self) -> None:
        """Write the current song list to the JSON index atomically."""
        tmp_path = self._json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self._songs], f, indent=2)
        os.replace(tmp_path, self._json_path)
