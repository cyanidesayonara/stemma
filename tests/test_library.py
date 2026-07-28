"""Tests for the song library."""

import json
import os
import shutil

import pytest

from src.library import SongLibrary, Song


@pytest.fixture
def library_dir(tmp_path):
    """Provide a temporary data directory for the library."""
    return str(tmp_path / "data")


@pytest.fixture
def library(library_dir):
    """Create a SongLibrary backed by a temporary directory."""
    return SongLibrary(data_dir=library_dir)


@pytest.fixture
def fake_audio(tmp_path):
    """Create a fake audio file to use as import source."""
    import numpy as np
    import soundfile as sf

    path = tmp_path / "test_song.wav"
    silence = np.zeros((44100, 2), dtype=np.float32)
    sf.write(str(path), silence, 44100)
    return str(path)


class TestSongDataclass:

    def test_song_has_required_fields(self):
        song = Song(
            id="abc123",
            title="Test Song",
            artist="Test Artist",
            original_path="test.wav",
            stems_path="/data/songs/abc123",
            model_used="htdemucs",
            date_added="2026-03-20T12:00:00",
        )
        assert song.id == "abc123"
        assert song.title == "Test Song"
        assert song.artist == "Test Artist"
        assert song.original_path == "test.wav"
        assert song.stems_path == "/data/songs/abc123"
        assert song.model_used == "htdemucs"
        assert song.date_added == "2026-03-20T12:00:00"

    def test_song_to_dict(self):
        song = Song(
            id="abc123",
            title="Test Song",
            artist="Test Artist",
            original_path="test.wav",
            stems_path="/data/songs/abc123",
            model_used="htdemucs",
            date_added="2026-03-20T12:00:00",
        )
        d = song.to_dict()
        assert d["id"] == "abc123"
        assert d["title"] == "Test Song"
        assert isinstance(d, dict)

    def test_song_from_dict(self):
        d = {
            "id": "abc123",
            "title": "Test Song",
            "artist": "Test Artist",
            "original_path": "test.wav",
            "stems_path": "/data/songs/abc123",
            "model_used": "htdemucs",
            "date_added": "2026-03-20T12:00:00",
        }
        song = Song.from_dict(d)
        assert song.id == "abc123"
        assert song.title == "Test Song"


class TestSongLibraryInit:

    def test_creates_data_directories(self, library, library_dir):
        assert os.path.isdir(library_dir)
        assert os.path.isdir(os.path.join(library_dir, "songs"))

    def test_creates_empty_library_json(self, library, library_dir):
        json_path = os.path.join(library_dir, "library.json")
        assert os.path.isfile(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data == []

    def test_loads_existing_library(self, library_dir):
        # Pre-populate library.json.
        song_dir = os.path.join(library_dir, "songs", "existing")
        os.makedirs(song_dir, exist_ok=True)
        json_path = os.path.join(library_dir, "library.json")
        existing = [{
            "id": "existing",
            "title": "Old Song",
            "artist": "Old Artist",
            "original_path": os.path.join(song_dir, "original.wav"),
            "stems_path": song_dir,
            "model_used": "htdemucs",
            "date_added": "2026-01-01T00:00:00",
        }]
        with open(json_path, "w") as f:
            json.dump(existing, f)

        lib = SongLibrary(data_dir=library_dir)
        assert len(lib.songs) == 1
        assert lib.songs[0].title == "Old Song"

    def test_recovers_from_corrupted_json(self, library_dir):
        """Corrupted library.json should not crash — starts empty instead."""
        os.makedirs(library_dir, exist_ok=True)
        json_path = os.path.join(library_dir, "library.json")
        with open(json_path, "w") as f:
            f.write("{invalid json content")

        lib = SongLibrary(data_dir=library_dir)
        assert len(lib.songs) == 0

        # The corrupted file should be overwritten with valid empty JSON.
        with open(json_path) as f:
            data = json.load(f)
        assert data == []

    def test_recovers_from_malformed_entries(self, library_dir):
        """Entries with missing fields should not crash the library."""
        os.makedirs(library_dir, exist_ok=True)
        json_path = os.path.join(library_dir, "library.json")
        with open(json_path, "w") as f:
            json.dump([{"bad": "data"}], f)

        lib = SongLibrary(data_dir=library_dir)
        assert len(lib.songs) == 0

    def test_corrupt_index_is_preserved_as_bak(self, library_dir):
        """A corrupt index must be kept for post-mortem, not destroyed."""
        os.makedirs(library_dir, exist_ok=True)
        json_path = os.path.join(library_dir, "library.json")
        with open(json_path, "w") as f:
            f.write("{not valid json")

        SongLibrary(data_dir=library_dir)
        bak = json_path + ".bak"
        assert os.path.isfile(bak)
        assert open(bak).read() == "{not valid json"

    def test_rebuilds_from_disk_when_index_corrupt(self, library_dir):
        """Songs whose dirs still hold an original.* file are recovered
        even when the index is unreadable."""
        songs_dir = os.path.join(library_dir, "songs")
        os.makedirs(os.path.join(songs_dir, "abc123"), exist_ok=True)
        with open(
            os.path.join(songs_dir, "abc123", "original.mp3"), "wb"
        ) as f:
            f.write(b"audio")
        # A dir with no original.* is skipped.
        os.makedirs(os.path.join(songs_dir, "empty"), exist_ok=True)

        json_path = os.path.join(library_dir, "library.json")
        with open(json_path, "w") as f:
            f.write("garbage")

        lib = SongLibrary(data_dir=library_dir)
        ids = {s.id for s in lib.songs}
        assert ids == {"abc123"}
        assert lib.songs[0].original_path.endswith("original.mp3")

    def test_recovers_from_non_utf8_index(self, library_dir):
        """A non-UTF-8 index (disk corruption) must not crash startup."""
        os.makedirs(library_dir, exist_ok=True)
        json_path = os.path.join(library_dir, "library.json")
        with open(json_path, "wb") as f:
            f.write(b"\xff\xfe\x00garbage")

        lib = SongLibrary(data_dir=library_dir)
        assert lib.songs == []
        assert os.path.isfile(json_path + ".bak")


class TestSongLibraryPathSafety:
    def test_rejects_loaded_traversal_path_without_touching_target(
        self, library_dir,
    ):
        songs_dir = os.path.join(library_dir, "songs")
        external_dir = os.path.join(library_dir, "external")
        os.makedirs(songs_dir, exist_ok=True)
        os.makedirs(external_dir)
        marker = os.path.join(external_dir, "keep.txt")
        with open(marker, "w") as f:
            f.write("keep")
        entry = {
            "id": "unsafe-traversal",
            "title": "Unsafe",
            "artist": "Artist",
            "original_path": os.path.join(external_dir, "original.wav"),
            "stems_path": os.path.join(songs_dir, "..", "external"),
            "model_used": "htdemucs",
            "date_added": "2026-01-01T00:00:00",
        }
        with open(os.path.join(library_dir, "library.json"), "w") as f:
            json.dump([entry], f)

        library = SongLibrary(library_dir)

        assert library.songs == []
        assert os.path.isfile(marker)

    def test_rejects_loaded_external_absolute_path_without_touching_target(
        self, library_dir, tmp_path,
    ):
        os.makedirs(os.path.join(library_dir, "songs"), exist_ok=True)
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        marker = external_dir / "keep.txt"
        marker.write_text("keep")
        entry = {
            "id": "unsafe-absolute",
            "title": "Unsafe",
            "artist": "Artist",
            "original_path": str(external_dir / "original.wav"),
            "stems_path": str(external_dir),
            "model_used": "htdemucs",
            "date_added": "2026-01-01T00:00:00",
        }
        with open(os.path.join(library_dir, "library.json"), "w") as f:
            json.dump([entry], f)

        library = SongLibrary(library_dir)

        assert library.songs == []
        assert marker.is_file()

    def test_remove_refuses_runtime_external_song_path(
        self, library, tmp_path,
    ):
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        marker = external_dir / "keep.txt"
        marker.write_text("keep")
        unsafe = Song(
            id="unsafe",
            title="Unsafe",
            artist="Artist",
            original_path=str(external_dir / "original.wav"),
            stems_path=str(external_dir),
            model_used="htdemucs",
            date_added="2026-01-01T00:00:00",
        )
        library._songs.append(unsafe)

        with pytest.raises(ValueError, match="outside"):
            library.remove_song(unsafe.id)

        assert library.get_song(unsafe.id) is unsafe
        assert marker.is_file()


class TestSongLibraryCRUD:

    def test_add_song(self, library, fake_audio):
        song = library.add_song(
            title="My Song",
            artist="My Artist",
            original_path=fake_audio,
        )
        assert song.title == "My Song"
        assert song.artist == "My Artist"
        assert song.id  # Non-empty ID generated
        assert len(library.songs) == 1

    def test_add_song_removes_partial_dir_when_copy_fails(
        self, library, fake_audio, monkeypatch
    ):
        songs_dir = library._songs_dir
        assert len(os.listdir(songs_dir)) == 0

        def fail_copy(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr("src.library.shutil.copy2", fail_copy)
        with pytest.raises(OSError, match="disk full"):
            library.add_song(
                title="T",
                artist="A",
                original_path=fake_audio,
            )
        assert len(library.songs) == 0
        assert len(os.listdir(songs_dir)) == 0

    def test_add_song_copies_audio_into_song_dir(self, library, library_dir, fake_audio):
        song = library.add_song(
            title="My Song",
            artist="My Artist",
            original_path=fake_audio,
        )
        # original_path should point inside the song directory, not the source.
        assert song.original_path.startswith(
            os.path.join(library_dir, "songs", song.id)
        )
        assert os.path.isfile(song.original_path)
        assert song.original_path.endswith(".wav")

    def test_add_song_preserves_source_extension(self, library, tmp_path):
        import numpy as np
        import soundfile as sf

        flac_path = tmp_path / "track.flac"
        sf.write(str(flac_path), np.zeros((44100, 2), dtype=np.float32), 44100)

        song = library.add_song(
            title="FLAC Song",
            artist="Artist",
            original_path=str(flac_path),
        )
        assert song.original_path.endswith(".flac")

    def test_add_song_creates_song_directory(self, library, library_dir, fake_audio):
        song = library.add_song(
            title="My Song",
            artist="My Artist",
            original_path=fake_audio,
        )
        song_dir = os.path.join(library_dir, "songs", song.id)
        assert os.path.isdir(song_dir)

    def test_add_song_persists_to_json(self, library, library_dir, fake_audio):
        library.add_song(
            title="My Song",
            artist="My Artist",
            original_path=fake_audio,
        )
        json_path = os.path.join(library_dir, "library.json")
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["title"] == "My Song"

    def test_get_song_by_id(self, library, fake_audio):
        song = library.add_song(
            title="Findable",
            artist="Artist",
            original_path=fake_audio,
        )
        found = library.get_song(song.id)
        assert found is not None
        assert found.title == "Findable"

    def test_get_song_not_found(self, library):
        assert library.get_song("nonexistent") is None

    def test_remove_song(self, library, library_dir, fake_audio):
        song = library.add_song(
            title="To Remove",
            artist="Artist",
            original_path=fake_audio,
        )
        song_dir = os.path.join(library_dir, "songs", song.id)
        assert os.path.isdir(song_dir)

        library.remove_song(song.id)
        assert len(library.songs) == 0
        assert not os.path.exists(song_dir)

    def test_remove_song_persists_to_json(self, library, library_dir, fake_audio):
        song = library.add_song(
            title="To Remove",
            artist="Artist",
            original_path=fake_audio,
        )
        library.remove_song(song.id)
        json_path = os.path.join(library_dir, "library.json")
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 0

    def test_remove_nonexistent_song_raises(self, library):
        with pytest.raises(KeyError):
            library.remove_song("nonexistent")

    def test_remove_save_failure_restores_entry_and_directory(
        self, library, fake_audio, monkeypatch,
    ):
        song = library.add_song(
            title="Keep Me",
            artist="Artist",
            original_path=fake_audio,
        )
        song_dir = song.stems_path

        def fail_save():
            raise OSError("index unavailable")

        monkeypatch.setattr(library, "_save", fail_save)

        with pytest.raises(OSError, match="index unavailable"):
            library.remove_song(song.id)

        assert library.get_song(song.id) is song
        assert os.path.isdir(song_dir)
        assert os.path.isfile(song.original_path)

    def test_update_song(self, library, fake_audio):
        song = library.add_song(
            title="Original",
            artist="Artist",
            original_path=fake_audio,
        )
        library.update_song(song.id, title="Updated Title")
        updated = library.get_song(song.id)
        assert updated.title == "Updated Title"
        assert updated.artist == "Artist"  # Unchanged

    def test_update_song_persists(self, library, library_dir, fake_audio):
        song = library.add_song(
            title="Original",
            artist="Artist",
            original_path=fake_audio,
        )
        library.update_song(song.id, model_used="htdemucs_6s")

        json_path = os.path.join(library_dir, "library.json")
        with open(json_path) as f:
            data = json.load(f)
        assert data[0]["model_used"] == "htdemucs_6s"

    def test_update_nonexistent_song_raises(self, library):
        with pytest.raises(KeyError):
            library.update_song("nonexistent", title="Nope")

    def test_update_only_changes_editable_metadata(self, library, fake_audio):
        song = library.add_song(
            title="Original",
            artist="Artist",
            original_path=fake_audio,
        )
        original_path = song.original_path
        stems_path = song.stems_path
        date_added = song.date_added

        library.update_song(
            song.id,
            title="Updated",
            artist="New Artist",
            model_used="htdemucs",
            original_path="C:/outside/original.wav",
            stems_path="C:/outside",
            date_added="changed",
            to_dict="changed",
        )

        assert song.title == "Updated"
        assert song.artist == "New Artist"
        assert song.model_used == "htdemucs"
        assert song.original_path == original_path
        assert song.stems_path == stems_path
        assert song.date_added == date_added
        assert callable(song.to_dict)

    def test_update_save_failure_restores_in_memory_values(
        self, library, fake_audio, monkeypatch,
    ):
        song = library.add_song(
            title="Original",
            artist="Artist",
            original_path=fake_audio,
        )

        def fail_save():
            raise OSError("index unavailable")

        monkeypatch.setattr(library, "_save", fail_save)

        with pytest.raises(OSError, match="index unavailable"):
            library.update_song(
                song.id,
                title="Changed",
                artist="Changed Artist",
            )

        assert song.title == "Original"
        assert song.artist == "Artist"

    def test_songs_property_returns_copy(self, library, fake_audio):
        library.add_song(title="Song", artist="A", original_path=fake_audio)
        songs = library.songs
        songs.clear()
        assert len(library.songs) == 1  # Internal list unaffected
