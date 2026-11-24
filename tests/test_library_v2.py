"""Tests for v3.2 versioned library schema v2, tags, practice status, and setlists."""

import glob
import json
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from src.library import (
    LIBRARY_SCHEMA_VERSION,
    PRACTICE_STATUSES,
    LibraryReadOnlyError,
    Setlist,
    SongLibrary,
    normalize_practice_status,
    normalize_tags,
)


@pytest.fixture
def library_dir(tmp_path):
    """Temporary data directory path."""
    return str(tmp_path / "data")


@pytest.fixture
def source_audio(tmp_path):
    """Deterministic short stereo WAV used as an import source."""
    path = tmp_path / "source.wav"
    sf.write(str(path), np.zeros((4410, 2), dtype=np.float32), 44100)
    return str(path)


@pytest.fixture
def populated_library(library_dir, source_audio):
    """Library pre-populated with three songs."""
    lib = SongLibrary(library_dir)
    s1 = lib.add_song("Song One", "Artist A", source_audio, "htdemucs")
    s2 = lib.add_song("Song Two", "Artist B", source_audio, "htdemucs")
    s3 = lib.add_song("Song Three", "Artist C", source_audio, "mdx_inst_hq3")
    return lib, (s1, s2, s3)


def _read_raw_index(data_dir: str) -> dict | list:
    with open(os.path.join(data_dir, "library.json"), encoding="utf-8") as f:
        return json.load(f)


def _make_song_dir(data_dir: str, song_id: str) -> str:
    song_dir = os.path.join(data_dir, "songs", song_id)
    os.makedirs(song_dir, exist_ok=True)
    with open(os.path.join(song_dir, "original.wav"), "wb") as f:
        f.write(b"audio")
    return song_dir


def _make_v1_entry(data_dir: str, song_id: str, title: str = "Track") -> dict:
    song_dir = _make_song_dir(data_dir, song_id)
    return {
        "id": song_id,
        "title": title,
        "artist": "Band",
        "original_path": os.path.join(song_dir, "original.wav"),
        "stems_path": song_dir,
        "model_used": "htdemucs",
        "date_added": "2026-01-01T00:00:00",
    }


class TestLibrarySchemaV2:
    """Versioned JSON schema v2 persistence, v1 migration, and recovery."""

    def test_fresh_library_writes_v2_envelope(self, library_dir):
        SongLibrary(library_dir)
        assert _read_raw_index(library_dir) == {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "songs": [],
            "setlists": [],
        }

    def test_v1_flat_list_migrates_cleanly_without_backup(self, library_dir):
        entry = _make_v1_entry(library_dir, "0123456789ab", title="Legacy Track")
        with open(os.path.join(library_dir, "library.json"), "w", encoding="utf-8") as f:
            json.dump([entry], f)

        lib = SongLibrary(library_dir)

        assert lib.recovery is None
        assert glob.glob(os.path.join(library_dir, "library.json*.bak")) == []
        assert len(lib.songs) == 1
        song = lib.songs[0]
        assert song.id == "0123456789ab"
        assert song.title == "Legacy Track"
        assert song.tags == []
        assert song.practice_status == ""
        assert lib.setlists == []

        migrated = _read_raw_index(library_dir)
        assert isinstance(migrated, dict)
        assert migrated["schema_version"] == 2
        assert len(migrated["songs"]) == 1
        assert migrated["setlists"] == []

    def test_future_schema_version_and_unknown_keys_survive(self, library_dir):
        entry = _make_v1_entry(library_dir, "0123456789ab", title="Future Song")
        entry["tags"] = ["gig"]
        entry["practice_status"] = "ready"
        entry["future_song_field"] = 42
        payload = {
            "schema_version": 3,
            "future_top_level": {"note": "ignore me"},
            "songs": [entry],
            "setlists": [
                {
                    "id": "sl1",
                    "name": "Friday Gig",
                    "song_ids": ["0123456789ab"],
                    "future_setlist_field": True,
                }
            ],
        }
        with open(os.path.join(library_dir, "library.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

        lib = SongLibrary(library_dir)

        assert lib.recovery is None
        assert [(s.id, s.tags, s.practice_status) for s in lib.songs] == [
            ("0123456789ab", ["gig"], "ready"),
        ]
        assert [(sl.id, sl.name, sl.song_ids) for sl in lib.setlists] == [
            ("sl1", "Friday Gig", ["0123456789ab"]),
        ]

    def test_unknown_practice_status_falls_back_to_empty(self, library_dir):
        entry = _make_v1_entry(library_dir, "0123456789ab")
        entry["practice_status"] = "mastered_in_v4"
        payload = {
            "schema_version": 2,
            "songs": [entry],
            "setlists": [],
        }
        with open(os.path.join(library_dir, "library.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

        lib = SongLibrary(library_dir)

        assert lib.recovery is None
        assert lib.songs[0].practice_status == ""

    @pytest.mark.parametrize(
        "bad_payload",
        [
            {"schema_version": "2", "songs": [], "setlists": []},
            {"schema_version": True, "songs": [], "setlists": []},
            {"schema_version": 0, "songs": [], "setlists": []},
            {"schema_version": 2, "songs": {}, "setlists": []},
            {"schema_version": 2, "songs": [], "setlists": {}},
        ],
    )
    def test_corrupt_v2_envelope_recovers_from_disk(self, library_dir, bad_payload):
        _make_song_dir(library_dir, "0123456789ab")
        with open(os.path.join(library_dir, "library.json"), "w", encoding="utf-8") as f:
            json.dump(bad_payload, f)

        lib = SongLibrary(library_dir)

        assert lib.recovery is not None
        assert lib.recovery.kind == "corrupt"
        assert os.path.isfile(lib.recovery.backup_path)
        assert [s.id for s in lib.songs] == ["0123456789ab"]
        assert lib.setlists == []

    def test_malformed_setlist_triggers_partial_recovery(self, library_dir):
        entry = _make_v1_entry(library_dir, "0123456789ab", title="Valid")
        payload = {
            "schema_version": 2,
            "songs": [entry],
            "setlists": [
                "not-a-setlist",
                {"id": "sl1", "name": "Good Set", "song_ids": ["0123456789ab"]},
            ],
        }
        with open(os.path.join(library_dir, "library.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

        lib = SongLibrary(library_dir)

        assert lib.recovery is not None
        assert lib.recovery.kind == "partial"
        assert os.path.isfile(lib.recovery.backup_path)
        assert [s.id for s in lib.songs] == ["0123456789ab"]
        assert [(sl.id, sl.name, sl.song_ids) for sl in lib.setlists] == [
            ("sl1", "Good Set", ["0123456789ab"]),
        ]

    def test_failed_partial_backup_on_bad_setlist_enters_unsaved_mode(
        self, library_dir, monkeypatch,
    ):
        entry = _make_v1_entry(library_dir, "0123456789ab", title="Valid")
        original = json.dumps({
            "schema_version": 2,
            "songs": [entry],
            "setlists": [{"id": "bad", "name": "   "}],
        })
        with open(os.path.join(library_dir, "library.json"), "w", encoding="utf-8") as f:
            f.write(original)

        monkeypatch.setattr(
            "src.library.shutil.copyfile",
            lambda *_a, **_k: (_ for _ in ()).throw(PermissionError(13, "denied")),
        )

        lib = SongLibrary(library_dir)

        assert not lib.writable
        assert lib.recovery is not None
        assert lib.recovery.kind == "unsaved"
        with open(os.path.join(library_dir, "library.json"), encoding="utf-8") as f:
            assert f.read() == original

    def test_stale_song_ids_pruned_while_recovered_and_held_preserved(
        self, library_dir, monkeypatch,
    ):
        # s1: healthy in index; s2: missing from songs list but present on disk
        # (recovered when partial backup runs); s3: unrestorable interrupted
        # removal kept in _held.
        e1 = _make_v1_entry(library_dir, "aaaaaaaaaaaa", title="Healthy")
        _make_song_dir(library_dir, "bbbbbbbbbbbb")
        e3 = _make_v1_entry(library_dir, "cccccccccccc", title="Held")
        staged = os.path.join(
            library_dir, "songs", f".remove-cccccccccccc-{'a' * 32}",
        )
        os.rename(e3["stems_path"], staged)

        real_replace = os.replace

        def lock_held(src, dst, *args, **kwargs):
            if ".remove-cccccccccccc-" in str(src):
                raise PermissionError(32, "in use")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr("src.library.os.replace", lock_held)

        payload = {
            "schema_version": 2,
            "songs": [
                e1,
                "corrupt-entry",  # triggers partial recovery + disk rebuild for s2
                e3,
            ],
            "setlists": [
                {
                    "id": "sl1",
                    "name": "Mixed",
                    "song_ids": [
                        "aaaaaaaaaaaa",
                        "stale-missing",
                        "bbbbbbbbbbbb",
                        "cccccccccccc",
                    ],
                }
            ],
        }
        with open(os.path.join(library_dir, "library.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

        lib = SongLibrary(library_dir)

        assert lib.setlists[0].song_ids == [
            "aaaaaaaaaaaa",
            "bbbbbbbbbbbb",
            "cccccccccccc",
        ]

    def test_moved_data_folder_preserves_tags_practice_status_and_setlists(
        self, tmp_path, source_audio,
    ):
        dir_a = str(tmp_path / "A")
        dir_b = str(tmp_path / "B")
        lib_a = SongLibrary(dir_a)
        s1 = lib_a.add_song("One", "Band", source_audio, "htdemucs")
        s2 = lib_a.add_song("Two", "Band", source_audio, "htdemucs")
        lib_a.set_song_tags(s1.id, ["gig", "opener"])
        lib_a.set_practice_status(s2.id, "learning")
        sl = lib_a.create_setlist("Saturday Show", [s2.id, s1.id])

        shutil.copytree(dir_a, dir_b)
        lib_b = SongLibrary(dir_b)

        assert lib_b.recovery is None
        assert lib_b.get_song(s1.id).tags == ["gig", "opener"]
        assert lib_b.get_song(s2.id).practice_status == "learning"
        assert lib_b.get_setlist(sl.id) == Setlist(
            id=sl.id, name="Saturday Show", song_ids=[s2.id, s1.id],
        )


class TestSongTags:
    """Tag normalization, single/batch mutations, rollback, and read-only guards."""

    def test_normalize_tags_cleans_and_deduplicates_case_insensitively(self):
        assert normalize_tags(["  gig ", "", "  ", "Gig", "learning", "GIG", "jazz"]) == [
            "gig", "learning", "jazz",
        ]

    @pytest.mark.parametrize("bad", ["gig", b"gig", 123, ["ok", 42]])
    def test_normalize_tags_rejects_invalid_types(self, bad):
        with pytest.raises(TypeError):
            normalize_tags(bad)

    def test_set_song_tags_persists_and_rolls_back(
        self, populated_library, library_dir, monkeypatch,
    ):
        lib, (s1, _s2, _s3) = populated_library
        lib.set_song_tags(s1.id, ["  Gig ", "gig", "Acoustic"])
        assert s1.tags == ["Gig", "Acoustic"]

        reloaded = SongLibrary(library_dir)
        assert reloaded.get_song(s1.id).tags == ["Gig", "Acoustic"]

        monkeypatch.setattr(
            lib, "_save",
            lambda: (_ for _ in ()).throw(OSError("disk full")),
        )
        with pytest.raises(OSError, match="disk full"):
            lib.set_song_tags(s1.id, ["Other"])
        assert s1.tags == ["Gig", "Acoustic"]

    def test_set_song_tags_unknown_and_read_only(self, populated_library):
        lib, (s1, _s2, _s3) = populated_library
        with pytest.raises(KeyError):
            lib.set_song_tags("missing", ["gig"])

        lib._writable = False
        with pytest.raises(LibraryReadOnlyError):
            lib.set_song_tags(s1.id, ["gig"])

    def test_add_and_remove_tag_batch_atomic_and_rollback(
        self, populated_library, library_dir, monkeypatch,
    ):
        lib, (s1, s2, s3) = populated_library
        lib.set_song_tags(s1.id, ["Gig"])

        save_calls = 0
        real_save = lib._save

        def counted_save():
            nonlocal save_calls
            save_calls += 1
            real_save()

        monkeypatch.setattr(lib, "_save", counted_save)
        lib.add_tag_to_songs([s1.id, s2.id], "  gig ")
        assert save_calls == 1
        assert s1.tags == ["Gig"]  # existing casing preserved
        assert s2.tags == ["gig"]
        assert s3.tags == []

        with pytest.raises(ValueError):
            lib.add_tag_to_songs([s1.id], "   ")
        with pytest.raises(KeyError):
            lib.add_tag_to_songs([s3.id, "missing"], "new")
        assert s3.tags == []  # no partial mutation

        monkeypatch.setattr(
            lib, "_save",
            lambda: (_ for _ in ()).throw(OSError("io fail")),
        )
        with pytest.raises(OSError, match="io fail"):
            lib.add_tag_to_songs([s2.id, s3.id], "lesson")
        assert s2.tags == ["gig"]
        assert s3.tags == []

        with pytest.raises(OSError, match="io fail"):
            lib.remove_tag_from_songs([s1.id, s2.id], "GIG")
        assert s1.tags == ["Gig"]
        assert s2.tags == ["gig"]

        monkeypatch.undo()
        lib.remove_tag_from_songs([s1.id, s2.id], "GIG")
        reloaded = SongLibrary(library_dir)
        assert reloaded.get_song(s1.id).tags == []
        assert reloaded.get_song(s2.id).tags == []

    def test_remove_tag_validation_and_missing_song(self, populated_library):
        lib, (s1, _s2, _s3) = populated_library
        lib.set_song_tags(s1.id, ["gig"])
        with pytest.raises(ValueError):
            lib.remove_tag_from_songs([s1.id], "")
        with pytest.raises(KeyError):
            lib.remove_tag_from_songs([s1.id, "missing"], "gig")
        assert s1.tags == ["gig"]

    def test_all_tags_sorted_case_insensitively(self, populated_library):
        lib, (s1, s2, s3) = populated_library
        lib.set_song_tags(s1.id, ["rock", "Acoustic"])
        lib.set_song_tags(s2.id, ["ROCK", "blues"])
        lib.set_song_tags(s3.id, ["Blues", "jazz"])
        assert lib.all_tags() == ["Acoustic", "blues", "jazz", "rock"]

    def test_update_song_supports_tags_and_practice_status_with_rollback(
        self, populated_library, monkeypatch,
    ):
        lib, (s1, _s2, _s3) = populated_library
        lib.update_song(s1.id, tags=["  Gig ", "gig"], practice_status=" LEARNING ")
        assert s1.tags == ["Gig"]
        assert s1.practice_status == "learning"

        monkeypatch.setattr(
            lib, "_save",
            lambda: (_ for _ in ()).throw(OSError("fail")),
        )
        with pytest.raises(OSError, match="fail"):
            lib.update_song(s1.id, tags=["Other"], practice_status="ready")
        assert s1.tags == ["Gig"]
        assert s1.practice_status == "learning"


class TestPracticeStatus:
    """Practice status validation, persistence, and rollback."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("", ""),
            ("  ", ""),
            (None, ""),
            ("learning", "learning"),
            (" PRACTICING ", "practicing"),
            ("Ready", "ready"),
        ],
    )
    def test_normalize_practice_status(self, raw, expected):
        assert normalize_practice_status(raw) == expected
        assert expected in PRACTICE_STATUSES

    def test_set_practice_status_persists_validates_and_rolls_back(
        self, populated_library, library_dir, monkeypatch,
    ):
        lib, (s1, _s2, _s3) = populated_library
        lib.set_practice_status(s1.id, "Practicing")
        assert s1.practice_status == "practicing"

        reloaded = SongLibrary(library_dir)
        assert reloaded.get_song(s1.id).practice_status == "practicing"

        with pytest.raises(ValueError):
            lib.set_practice_status(s1.id, "finished")
        with pytest.raises(TypeError):
            lib.set_practice_status(s1.id, 1)
        with pytest.raises(KeyError):
            lib.set_practice_status("missing", "ready")

        monkeypatch.setattr(
            lib, "_save",
            lambda: (_ for _ in ()).throw(OSError("fail")),
        )
        with pytest.raises(OSError, match="fail"):
            lib.set_practice_status(s1.id, "ready")
        assert s1.practice_status == "practicing"

        lib._writable = False
        with pytest.raises(LibraryReadOnlyError):
            lib.set_practice_status(s1.id, "ready")


class TestSetlists:
    """Ordered setlists CRUD, reordering, defensive copies, and song removal."""

    def test_create_rename_delete_and_persist(
        self, populated_library, library_dir, monkeypatch,
    ):
        lib, (s1, s2, _s3) = populated_library
        sl = lib.create_setlist("  Lesson 1 ", [s1.id, s2.id, s1.id])
        assert sl.name == "Lesson 1"
        assert sl.song_ids == [s1.id, s2.id]

        reloaded = SongLibrary(library_dir)
        assert reloaded.get_setlist(sl.id) == sl

        renamed = lib.rename_setlist(sl.id, "  Recital ")
        assert renamed.name == "Recital"
        assert SongLibrary(library_dir).get_setlist(sl.id).name == "Recital"

        with pytest.raises(ValueError):
            lib.create_setlist("   ")
        with pytest.raises(KeyError):
            lib.create_setlist("Bad", ["missing"])
        with pytest.raises(ValueError):
            lib.rename_setlist(sl.id, "")
        with pytest.raises(KeyError):
            lib.rename_setlist("missing", "Valid")

        monkeypatch.setattr(
            lib, "_save",
            lambda: (_ for _ in ()).throw(OSError("fail")),
        )
        with pytest.raises(OSError, match="fail"):
            lib.create_setlist("Rolled Back")
        assert len(lib.setlists) == 1

        with pytest.raises(OSError, match="fail"):
            lib.rename_setlist(sl.id, "Rolled Back")
        assert lib.get_setlist(sl.id).name == "Recital"

        with pytest.raises(OSError, match="fail"):
            lib.delete_setlist(sl.id)
        assert len(lib.setlists) == 1

        monkeypatch.undo()
        lib.delete_setlist(sl.id)
        assert lib.setlists == []
        assert SongLibrary(library_dir).setlists == []
        with pytest.raises(KeyError):
            lib.delete_setlist(sl.id)

    def test_add_remove_and_reorder_songs(
        self, populated_library, library_dir, monkeypatch,
    ):
        lib, (s1, s2, s3) = populated_library
        sl = lib.create_setlist("Gig", [s1.id])

        updated = lib.add_songs_to_setlist(sl.id, [s2.id, s1.id, s3.id])
        assert updated.song_ids == [s1.id, s2.id, s3.id]

        with pytest.raises(KeyError):
            lib.add_songs_to_setlist(sl.id, ["missing"])
        with pytest.raises(KeyError):
            lib.add_songs_to_setlist("missing", [s1.id])

        reordered = lib.reorder_setlist(sl.id, [s3.id, s1.id, s2.id])
        assert reordered.song_ids == [s3.id, s1.id, s2.id]
        assert SongLibrary(library_dir).get_setlist(sl.id).song_ids == [
            s3.id, s1.id, s2.id,
        ]

        # Non-permutations must fail.
        for bad_order in (
            [s1.id, s2.id],  # missing s3
            [s1.id, s1.id, s2.id],  # duplicate
            [s1.id, s2.id, s3.id, s3.id],  # extra duplicate
            [s1.id, s2.id, "other"],  # wrong song
        ):
            with pytest.raises(ValueError):
                lib.reorder_setlist(sl.id, bad_order)
        assert lib.get_setlist(sl.id).song_ids == [s3.id, s1.id, s2.id]

        monkeypatch.setattr(
            lib, "_save",
            lambda: (_ for _ in ()).throw(OSError("fail")),
        )
        with pytest.raises(OSError, match="fail"):
            lib.reorder_setlist(sl.id, [s1.id, s2.id, s3.id])
        assert lib.get_setlist(sl.id).song_ids == [s3.id, s1.id, s2.id]

        with pytest.raises(OSError, match="fail"):
            lib.remove_song_from_setlist(sl.id, s1.id)
        assert lib.get_setlist(sl.id).song_ids == [s3.id, s1.id, s2.id]

        monkeypatch.undo()
        after_remove = lib.remove_song_from_setlist(sl.id, s1.id)
        assert after_remove.song_ids == [s3.id, s2.id]
        with pytest.raises(KeyError):
            lib.remove_song_from_setlist("missing", s2.id)

    def test_defensive_copies_protect_internal_state(self, populated_library):
        lib, (s1, s2, _s3) = populated_library
        sl = lib.create_setlist("Gig", [s1.id])
        sl.song_ids.append(s2.id)
        lib.setlists[0].song_ids.clear()
        fetched = lib.get_setlist(sl.id)
        fetched.name = "Mutated"
        fetched.song_ids.append(s2.id)

        assert lib.get_setlist(sl.id) == Setlist(
            id=sl.id, name="Gig", song_ids=[s1.id],
        )

    def test_remove_song_prunes_setlists_and_rolls_back_on_save_error(
        self, populated_library, library_dir, monkeypatch,
    ):
        lib, (s1, s2, s3) = populated_library
        sl_a = lib.create_setlist("Set A", [s1.id, s2.id])
        sl_b = lib.create_setlist("Set B", [s2.id, s3.id])

        monkeypatch.setattr(
            lib, "_save",
            lambda: (_ for _ in ()).throw(OSError("index locked")),
        )
        with pytest.raises(OSError, match="index locked"):
            lib.remove_song(s2.id)
        assert lib.get_song(s2.id) is not None
        assert lib.get_setlist(sl_a.id).song_ids == [s1.id, s2.id]
        assert lib.get_setlist(sl_b.id).song_ids == [s2.id, s3.id]

        monkeypatch.undo()
        lib.remove_song(s2.id)
        assert lib.get_setlist(sl_a.id).song_ids == [s1.id]
        assert lib.get_setlist(sl_b.id).song_ids == [s3.id]

        reloaded = SongLibrary(library_dir)
        assert reloaded.get_setlist(sl_a.id).song_ids == [s1.id]
        assert reloaded.get_setlist(sl_b.id).song_ids == [s3.id]

    def test_setlist_mutators_enforce_read_only(self, populated_library):
        lib, (s1, s2, _s3) = populated_library
        sl = lib.create_setlist("Gig", [s1.id])
        lib._writable = False

        with pytest.raises(LibraryReadOnlyError):
            lib.create_setlist("New")
        with pytest.raises(LibraryReadOnlyError):
            lib.rename_setlist(sl.id, "Renamed")
        with pytest.raises(LibraryReadOnlyError):
            lib.delete_setlist(sl.id)
        with pytest.raises(LibraryReadOnlyError):
            lib.add_songs_to_setlist(sl.id, [s2.id])
        with pytest.raises(LibraryReadOnlyError):
            lib.remove_song_from_setlist(sl.id, s1.id)
        with pytest.raises(LibraryReadOnlyError):
            lib.reorder_setlist(sl.id, [s1.id])

    def test_empty_string_song_id_raises_key_error_in_mutators(
        self, populated_library,
    ):
        lib, (s1, _s2, _s3) = populated_library
        sl = lib.create_setlist("Gig", [s1.id])

        with pytest.raises(KeyError):
            lib.create_setlist("Bad", [""])
        with pytest.raises(KeyError):
            lib.add_songs_to_setlist(sl.id, [""])
        with pytest.raises(KeyError):
            lib.add_tag_to_songs([""], "rock")
        with pytest.raises(KeyError):
            lib.remove_tag_from_songs([""], "rock")
        assert Setlist.from_dict(
            {"id": "sl2", "name": "Tolerant", "song_ids": ["", s1.id, ""]},
        ).song_ids == [s1.id]
