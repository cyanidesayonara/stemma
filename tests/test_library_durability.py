"""Library durability: moved data folders, locked or corrupt indexes,
and a startup prune that cannot stop the app from opening."""

import builtins
import glob
import json
import os
import shutil
from unittest.mock import MagicMock

import pytest

from src import library as library_module
from src.library import SongLibrary


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Keep the read retries instant."""
    monkeypatch.setattr(library_module, "_sleep", lambda _s: None)


@pytest.fixture
def source_audio(tmp_path):
    path = tmp_path / "source.wav"
    path.write_bytes(b"audio")
    return str(path)


def _index(data_dir):
    with open(os.path.join(data_dir, "library.json"), encoding="utf-8") as f:
        return json.load(f)


def _under(path, root):
    path = os.path.normcase(os.path.realpath(path))
    root = os.path.normcase(os.path.realpath(root))
    return os.path.commonpath([path, root]) == root


class TestMovedDataFolder:

    def test_copied_data_folder_keeps_songs_and_metadata(
        self, tmp_path, source_audio,
    ):
        dir_a = str(tmp_path / "A")
        dir_b = str(tmp_path / "B")
        lib_a = SongLibrary(dir_a)
        song = lib_a.add_song("Title", "Artist", source_audio, "htdemucs")
        shutil.copytree(dir_a, dir_b)

        lib_b = SongLibrary(dir_b)

        (moved,) = lib_b.songs
        assert (moved.id, moved.title, moved.artist, moved.model_used) == (
            song.id, "Title", "Artist", "htdemucs",
        )
        assert _under(moved.stems_path, dir_b)
        assert _under(moved.original_path, dir_b)
        assert os.path.isfile(moved.original_path)
        assert lib_b.recovery is None

    def test_saving_after_a_move_keeps_the_song(
        self, tmp_path, source_audio,
    ):
        dir_a = str(tmp_path / "A")
        dir_b = str(tmp_path / "B")
        lib_a = SongLibrary(dir_a)
        lib_a.add_song("Title", "Artist", source_audio, "htdemucs")
        shutil.move(dir_a, dir_b)

        lib_b = SongLibrary(dir_b)
        lib_b.add_song("Second", "Band", source_audio)

        reloaded = SongLibrary(dir_b)
        assert sorted(s.title for s in reloaded.songs) == ["Second", "Title"]
        for entry in _index(dir_b):
            assert _under(entry["stems_path"], dir_b)
            assert _under(entry["original_path"], dir_b)

    def test_song_outside_root_without_a_folder_is_backed_up(
        self, tmp_path,
    ):
        """An entry that cannot be rebased is left out, but the index that
        named it is kept before anything is written."""
        data_dir = str(tmp_path / "data")
        os.makedirs(os.path.join(data_dir, "songs"))
        entry = {
            "id": "0123456789ab", "title": "Gone", "artist": "Band",
            "original_path": str(tmp_path / "elsewhere" / "original.wav"),
            "stems_path": str(tmp_path / "elsewhere"),
            "model_used": "htdemucs", "date_added": "2026-01-01",
        }
        with open(os.path.join(data_dir, "library.json"), "w") as f:
            json.dump([entry], f)

        lib = SongLibrary(data_dir)

        assert lib.songs == []
        backups = glob.glob(os.path.join(data_dir, "library.json.*.bak"))
        assert len(backups) == 1
        with open(backups[0], encoding="utf-8") as f:
            assert json.load(f)[0]["title"] == "Gone"

    @pytest.mark.parametrize("song_id", ["..", ".", ".remove-x", "a/b"])
    def test_unsafe_ids_are_never_rebased(self, tmp_path, song_id):
        data_dir = str(tmp_path / "data")
        os.makedirs(os.path.join(data_dir, "songs", "a", "b"))
        entry = {
            "id": song_id, "title": "Unsafe", "artist": "",
            "original_path": "", "stems_path": str(tmp_path / "elsewhere"),
            "model_used": "", "date_added": "",
        }
        with open(os.path.join(data_dir, "library.json"), "w") as f:
            json.dump([entry], f)

        assert SongLibrary(data_dir).songs == []

    def test_legacy_migration_keeps_the_library(
        self, tmp_path, source_audio, monkeypatch,
    ):
        """The one-time copy of a repo data/ tree carries absolute paths
        into the new folder."""
        from src import data_paths

        app_root = tmp_path / "app"
        legacy = app_root / "data"
        lib = SongLibrary(str(legacy))
        lib.add_song("Old", "Band", source_audio, "htdemucs")
        dest = tmp_path / "user"
        monkeypatch.setattr(
            data_paths, "platform_user_data_dir", lambda: str(dest),
        )
        settings = MagicMock()
        settings.value.side_effect = lambda key, default=None, **_kw: default

        resolved = data_paths.resolve_data_dir(str(app_root), settings)

        (song,) = SongLibrary(resolved).songs
        assert song.title == "Old"
        assert _under(song.stems_path, str(dest))


def _song_on_disk(data_dir, song_id="0123456789ab", title="Kept"):
    song_dir = os.path.join(data_dir, "songs", song_id)
    os.makedirs(song_dir)
    open(os.path.join(song_dir, "original.wav"), "wb").close()
    entry = {
        "id": song_id, "title": title, "artist": "Band",
        "original_path": os.path.join(song_dir, "original.wav"),
        "stems_path": song_dir, "model_used": "htdemucs",
        "date_added": "2026-01-01",
    }
    return entry


def _write_index(data_dir, text):
    with open(os.path.join(data_dir, "library.json"), "w",
              encoding="utf-8") as f:
        f.write(text)


def _fail_opening_index(monkeypatch, data_dir, times):
    """Make reading library.json raise PermissionError *times* times."""
    target = os.path.normcase(os.path.join(data_dir, "library.json"))
    real_open = builtins.open
    calls = {"n": 0}

    def fake_open(file, mode="r", *args, **kwargs):
        if (
            isinstance(file, str)
            and os.path.normcase(file) == target
            and "r" in mode
            and calls["n"] < times
        ):
            calls["n"] += 1
            raise PermissionError(13, "The process cannot access the file")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return calls


class TestUnreadableIndex:

    def test_transient_lock_is_retried(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "data")
        entry = _song_on_disk(data_dir)
        _write_index(data_dir, json.dumps([entry]))
        calls = _fail_opening_index(monkeypatch, data_dir, times=2)

        lib = SongLibrary(data_dir)

        assert calls["n"] == 2
        assert [(s.title, s.artist) for s in lib.songs] == [("Kept", "Band")]
        assert lib.recovery is None
        assert glob.glob(os.path.join(data_dir, "library.json*.bak")) == []

    def test_persistent_lock_leaves_the_index_untouched(
        self, tmp_path, monkeypatch,
    ):
        data_dir = str(tmp_path / "data")
        entry = _song_on_disk(data_dir)
        original = json.dumps([entry])
        _write_index(data_dir, original)
        _fail_opening_index(monkeypatch, data_dir, times=1000)

        lib = SongLibrary(data_dir)

        # The folders are listed so the session is usable ...
        assert [s.id for s in lib.songs] == ["0123456789ab"]
        assert lib.recovery is not None
        assert lib.recovery.kind == "unreadable"
        assert not lib.writable
        # ... but nothing is written over the index.
        with pytest.raises(OSError):
            lib.update_song("0123456789ab", title="Edited")
        monkeypatch.undo()
        with open(os.path.join(data_dir, "library.json"),
                  encoding="utf-8") as f:
            assert f.read() == original
        assert glob.glob(os.path.join(data_dir, "library.json*.bak")) == []

    def test_save_failure_during_load_is_not_corruption(
        self, tmp_path, monkeypatch,
    ):
        """A failed write while tidying a readable index must not send the
        index down the corrupt path."""
        data_dir = str(tmp_path / "data")
        entry = _song_on_disk(data_dir)
        original = json.dumps(["not a song", entry])
        _write_index(data_dir, original)

        def failing_save(self):
            raise PermissionError(13, "locked")

        monkeypatch.setattr(SongLibrary, "_save", failing_save)

        lib = SongLibrary(data_dir)

        assert [s.title for s in lib.songs] == ["Kept"]
        with open(os.path.join(data_dir, "library.json"),
                  encoding="utf-8") as f:
            assert f.read() == original


class TestCorruptIndex:

    def test_corrupt_twice_keeps_two_distinct_backups(self, tmp_path):
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)
        _write_index(data_dir, "{first damage")
        first = SongLibrary(data_dir)
        assert first.recovery is not None
        assert first.recovery.kind == "corrupt"
        _write_index(data_dir, "{second damage")

        second = SongLibrary(data_dir)

        backups = sorted(
            glob.glob(os.path.join(data_dir, "library.json*.bak"))
        )
        assert len(backups) == 2
        contents = set()
        for path in backups:
            with open(path, encoding="utf-8") as f:
                contents.add(f.read())
        assert contents == {"{first damage", "{second damage"}
        assert first.recovery.backup_path != second.recovery.backup_path
        assert os.path.isfile(second.recovery.backup_path)

    def test_recovery_message_names_the_backup(self, tmp_path):
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)
        _write_index(data_dir, "{damage")

        lib = SongLibrary(data_dir)

        message = lib.recovery.message
        backup = lib.recovery.backup_path
        assert os.path.basename(backup) in message
        assert os.path.dirname(backup) in message
        assert "re-enter" in message.lower()

    def test_partial_damage_reports_recovery(self, tmp_path):
        data_dir = str(tmp_path / "data")
        entry = _song_on_disk(data_dir)
        _write_index(data_dir, json.dumps(["not a song", entry]))

        lib = SongLibrary(data_dir)

        assert lib.recovery is not None
        assert os.path.isfile(lib.recovery.backup_path)

    def test_healthy_index_reports_nothing(self, tmp_path):
        data_dir = str(tmp_path / "data")
        entry = _song_on_disk(data_dir)
        _write_index(data_dir, json.dumps([entry]))

        assert SongLibrary(data_dir).recovery is None


class TestPruneCannotBlockStartup:

    def _library_with(self, tmp_path, source_audio, count):
        library = SongLibrary(str(tmp_path / "data"))
        return library, [
            library.add_song(f"Cut {n}", "Band", source_audio)
            for n in range(count)
        ]

    def test_locked_folder_does_not_stop_the_prune(
        self, tmp_path, source_audio, monkeypatch,
    ):
        from src.ui.main_window import MainWindow

        library, songs = self._library_with(tmp_path, source_audio, 3)
        locked = songs[0]
        real_replace = os.replace

        def replace(src, dst, *args, **kwargs):
            if os.path.normcase(str(src)) == os.path.normcase(
                locked.stems_path
            ):
                raise PermissionError(32, "in use")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(library_module.os, "replace", replace)
        stub = MagicMock()
        stub._library = library

        MainWindow._prune_incomplete_songs(stub)

        assert [s.id for s in library.songs] == [locked.id]
        for song in songs[1:]:
            assert not os.path.exists(song.stems_path)

    def test_failed_folder_delete_after_save_does_not_raise(
        self, tmp_path, source_audio, monkeypatch,
    ):
        library, (song, other) = self._library_with(
            tmp_path, source_audio, 2,
        )

        def rmtree(path, *args, **kwargs):
            raise PermissionError(32, "in use")

        monkeypatch.setattr(library_module.shutil, "rmtree", rmtree)

        library.remove_song(song.id)

        assert [s.id for s in library.songs] == [other.id]
        assert [e["id"] for e in _index(library._data_dir)] == [other.id]

    def test_leftover_staged_removal_is_cleaned_by_the_next_prune(
        self, tmp_path, source_audio, monkeypatch,
    ):
        from src.ui.main_window import MainWindow

        library, (song, kept) = self._library_with(tmp_path, source_audio, 2)
        real_rmtree = shutil.rmtree
        monkeypatch.setattr(
            library_module.shutil, "rmtree",
            lambda *a, **k: (_ for _ in ()).throw(PermissionError(32, "x")),
        )
        library.remove_song(song.id)
        songs_dir = os.path.join(library._data_dir, "songs")
        assert any(n.startswith(".remove-") for n in os.listdir(songs_dir))
        monkeypatch.setattr(library_module.shutil, "rmtree", real_rmtree)
        # Finish the kept song so the prune leaves it alone.
        from src.separation_state import write_completion_marker
        for stem in ("vocals", "other"):
            open(os.path.join(kept.stems_path, f"{stem}.wav"), "wb").close()
        write_completion_marker(kept.stems_path, "mdx_inst_hq3")

        relaunched = SongLibrary(library._data_dir)
        stub = MagicMock()
        stub._library = relaunched
        MainWindow._prune_incomplete_songs(stub)

        assert not any(
            n.startswith(".remove-") for n in os.listdir(songs_dir)
        )
        assert [s.id for s in relaunched.songs] == [kept.id]

    def test_prune_survives_remove_song_raising(
        self, tmp_path, source_audio, monkeypatch,
    ):
        from src.ui.main_window import MainWindow

        library, (song,) = self._library_with(tmp_path, source_audio, 1)
        monkeypatch.setattr(
            library, "remove_song",
            MagicMock(side_effect=PermissionError(32, "in use")),
        )
        stub = MagicMock()
        stub._library = library

        MainWindow._prune_incomplete_songs(stub)  # must not raise

        library.remove_song.assert_called_once_with(song.id)

    def test_read_only_library_is_not_pruned(self):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._library.writable = False
        stub._library.songs = [MagicMock()]

        MainWindow._prune_incomplete_songs(stub)

        stub._library.remove_song.assert_not_called()


class TestRecoveryNotice:

    def _show(self, monkeypatch, library):
        from src.ui import main_window

        monkeypatch.delenv("CI", raising=False)
        box = MagicMock()
        monkeypatch.setattr(main_window, "QMessageBox", box)
        stub = MagicMock()
        stub._library = library
        main_window.MainWindow._maybe_show_library_recovery_notice(stub)
        return box

    def test_recovered_library_shows_the_backup_path(
        self, tmp_path, monkeypatch,
    ):
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)
        _write_index(data_dir, "{damage")
        library = SongLibrary(data_dir)

        box = self._show(monkeypatch, library)

        box.warning.assert_called_once()
        text = box.warning.call_args.args[2]
        assert text == library.recovery.message

    def test_healthy_or_mock_library_shows_nothing(
        self, tmp_path, monkeypatch,
    ):
        box = self._show(monkeypatch, SongLibrary(str(tmp_path / "data")))
        box.warning.assert_not_called()
        box = self._show(monkeypatch, MagicMock())
        box.warning.assert_not_called()


def test_data_folder_change_says_songs_are_not_moved():
    from src.ui.preferences_dialog import data_dir_change_message

    text = data_dir_change_message(r"C:\old\stemma")
    assert "not moved" in text
    assert r"C:\old\stemma" in text


class TestHeldStagedRemoval:
    """A removal interrupted before its index save leaves the song's only
    copy in songs/.remove-<id>-<hex>. If that cannot be restored at
    startup, it must survive every later save and prune."""

    def _interrupted_removal(self, tmp_path, source_audio):
        data_dir = str(tmp_path / "data")
        lib = SongLibrary(data_dir)
        song = lib.add_song("Real Title", "Real Artist", source_audio,
                            "htdemucs")
        staged = os.path.join(
            data_dir, "songs", f".remove-{song.id}-{'a' * 32}",
        )
        os.rename(song.stems_path, staged)
        return data_dir, song, staged

    def _lock_staged(self, monkeypatch):
        real_replace = os.replace

        def replace(src, dst, *args, **kwargs):
            if ".remove-" in str(src):
                raise PermissionError(32, "in use")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(library_module.os, "replace", replace)
        return real_replace

    def test_unrestorable_staged_folder_survives_prune_and_saves(
        self, tmp_path, source_audio, monkeypatch,
    ):
        from src.ui.main_window import MainWindow

        data_dir, song, staged = self._interrupted_removal(
            tmp_path, source_audio,
        )
        real_replace = self._lock_staged(monkeypatch)
        lib = SongLibrary(data_dir)
        monkeypatch.setattr(library_module.os, "replace", real_replace)

        stub = MagicMock()
        stub._library = lib
        MainWindow._prune_incomplete_songs(stub)
        lib.add_song("Other", "Band", source_audio)

        assert os.path.isdir(staged)
        held = [e for e in _index(data_dir) if e["id"] == song.id]
        assert [e["title"] for e in held] == ["Real Title"]

        # Once the lock is gone, the next launch restores it.
        relaunched = SongLibrary(data_dir)
        restored = relaunched.get_song(song.id)
        assert restored is not None
        assert restored.title == "Real Title"
        assert os.path.isdir(restored.stems_path)

    def test_staged_folder_named_in_the_index_is_never_discarded(
        self, tmp_path, source_audio, monkeypatch,
    ):
        data_dir, song, staged = self._interrupted_removal(
            tmp_path, source_audio,
        )
        self._lock_staged(monkeypatch)
        lib = SongLibrary(data_dir)

        lib.discard_staged_removals()

        assert os.path.isdir(staged)


class TestRecoveryEdgeCases:

    def test_failed_partial_backup_has_its_own_message(
        self, tmp_path, monkeypatch,
    ):
        data_dir = str(tmp_path / "data")
        entry = _song_on_disk(data_dir)
        original = json.dumps(["not a song", entry])
        _write_index(data_dir, original)

        def copyfile(*_a, **_k):
            raise PermissionError(13, "denied")

        monkeypatch.setattr(library_module.shutil, "copyfile", copyfile)

        lib = SongLibrary(data_dir)

        assert [s.title for s in lib.songs] == ["Kept"]
        assert not lib.writable
        assert lib.recovery.kind == "unsaved"
        assert "folder names" not in lib.recovery.message
        with open(os.path.join(data_dir, "library.json"),
                  encoding="utf-8") as f:
            assert f.read() == original

    def test_missing_index_is_rebuilt_from_song_folders(self, tmp_path):
        """A corrupt index moved aside whose rewrite failed leaves no
        library.json; the next launch must not write an empty one."""
        data_dir = str(tmp_path / "data")
        _song_on_disk(data_dir)

        lib = SongLibrary(data_dir)

        assert [s.id for s in lib.songs] == ["0123456789ab"]
        assert [e["id"] for e in _index(data_dir)] == ["0123456789ab"]
        assert lib.recovery.kind == "missing"
        assert "re-enter" in lib.recovery.message.lower()

    def test_fresh_data_folder_reports_nothing(self, tmp_path):
        lib = SongLibrary(str(tmp_path / "data"))

        assert lib.songs == []
        assert lib.recovery is None


def test_failed_import_rollback_survives_a_locked_folder(monkeypatch):
    from src.ui import main_window
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(main_window, "QMessageBox", MagicMock())
    stub = MagicMock()
    stub._library.remove_song.side_effect = PermissionError(32, "in use")

    MainWindow._on_separation_failed(stub, "abc", "Separation failed")

    stub._library.remove_song.assert_called_once_with("abc")
    stub._library_panel.refresh.assert_called()
