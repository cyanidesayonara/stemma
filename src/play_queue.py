"""Play queue state machine and setlist autoplay coordinator.

Keeps playback ordering ("Up Next", active setlist context, shuffle pool,
and navigation history) separate from the library collection view so users can
browse, search, and manage songs without interrupting what plays next.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import random

CONTEXT_LIBRARY = "library"
CONTEXT_SETLIST = "setlist"

REPEAT_OFF = "off"
REPEAT_ALL = "all"
REPEAT_ONE = "one"
_VALID_REPEAT_MODES = frozenset({REPEAT_OFF, REPEAT_ALL, REPEAT_ONE})

_MAX_HISTORY = 100


def _validate_song_id(song_id: str) -> str:
    """Validate that *song_id* is a non-empty string."""
    if not isinstance(song_id, str) or not song_id.strip():
        raise ValueError("Song ID must be a non-empty string")
    return song_id


def _validate_song_ids(song_ids: Iterable[str]) -> list[str]:
    """Convert *song_ids* into a validated list without partial mutation."""
    if isinstance(song_ids, (str, bytes)):
        raise ValueError("song_ids must be an iterable of song ID strings")
    try:
        items = list(song_ids)
    except TypeError as exc:
        raise ValueError("song_ids must be an iterable of song ID strings") from exc
    return [_validate_song_id(sid) for sid in items]


class PlayQueue:
    """In-memory and serializable playback queue with setlist context."""

    def __init__(self) -> None:
        self._up_next: list[str] = []
        self._context_kind: str = CONTEXT_LIBRARY
        self._context_id: str | None = None
        self._context_name: str = ""
        self._context_songs: list[str] = []
        self._context_index: int = -1
        self._history: list[str] = []
        self._shuffle_queue: list[str] = []
        self._shuffle_visited: set[str] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def up_next(self) -> list[str]:
        """Explicitly queued song IDs (played before context tracks)."""
        return list(self._up_next)

    @property
    def context_kind(self) -> str:
        return self._context_kind

    @property
    def context_id(self) -> str | None:
        return self._context_id

    @property
    def context_name(self) -> str:
        return self._context_name

    @property
    def context_songs(self) -> list[str]:
        return list(self._context_songs)

    @property
    def context_index(self) -> int:
        return self._context_index

    @property
    def history(self) -> list[str]:
        return list(self._history)

    @property
    def has_up_next(self) -> bool:
        return bool(self._up_next)

    @property
    def is_setlist_active(self) -> bool:
        return (
            self._context_kind == CONTEXT_SETLIST
            and bool(self._context_id)
            and bool(self._context_songs)
        )

    def __len__(self) -> int:
        return len(self._up_next)

    # ------------------------------------------------------------------
    # Explicit queue mutators
    # ------------------------------------------------------------------

    def play_next(self, song_id: str) -> None:
        """Insert *song_id* at the front of Up Next."""
        clean = _validate_song_id(song_id)
        self._up_next.insert(0, clean)

    def play_next_many(self, song_ids: Iterable[str]) -> None:
        """Insert *song_ids* at the front of Up Next, preserving order."""
        items = _validate_song_ids(song_ids)
        if items:
            self._up_next = items + self._up_next

    def enqueue(self, song_id: str) -> None:
        """Append *song_id* to the end of Up Next."""
        clean = _validate_song_id(song_id)
        self._up_next.append(clean)

    def enqueue_many(self, song_ids: Iterable[str]) -> None:
        """Append *song_ids* to the end of Up Next, preserving order."""
        items = _validate_song_ids(song_ids)
        if items:
            self._up_next.extend(items)

    def remove_at(self, index: int) -> str:
        """Remove and return the queued item at *index*."""
        if index < 0 or index >= len(self._up_next):
            raise IndexError(f"Queue index {index} out of range")
        return self._up_next.pop(index)

    def move(self, from_index: int, to_index: int) -> None:
        """Move a queued item from *from_index* to *to_index*."""
        length = len(self._up_next)
        if not (0 <= from_index < length) or not (0 <= to_index < length):
            raise IndexError(
                f"Queue move ({from_index} -> {to_index}) out of range"
            )
        if from_index == to_index:
            return
        item = self._up_next.pop(from_index)
        self._up_next.insert(to_index, item)

    def clear_up_next(self) -> None:
        """Clear Up Next while keeping the active playback context."""
        self._up_next.clear()

    # ------------------------------------------------------------------
    # Setlist playback context
    # ------------------------------------------------------------------

    def start_setlist(
        self,
        setlist_id: str,
        song_ids: Iterable[str],
        *,
        name: str = "",
        start_index: int = 0,
        clear_up_next: bool = False,
    ) -> str | None:
        """Activate a setlist context and return its initial song ID."""
        if not isinstance(setlist_id, str) or not setlist_id.strip():
            raise ValueError("Setlist ID must be a non-empty string")
        songs = _validate_song_ids(song_ids)
        if clear_up_next:
            self._up_next.clear()
        if not songs:
            self.clear_setlist()
            return None
        if start_index < 0 or start_index >= len(songs):
            raise IndexError(f"Setlist start_index {start_index} out of range")

        self._context_kind = CONTEXT_SETLIST
        self._context_id = setlist_id
        self._context_name = str(name) if isinstance(name, str) else ""
        self._context_songs = songs
        self._context_index = start_index
        self._shuffle_queue.clear()
        self._shuffle_visited = {songs[start_index]}
        return songs[start_index]

    def sync_setlist(
        self,
        setlist_id: str,
        song_ids: Iterable[str],
        *,
        name: str | None = None,
        current_song_id: str | None = None,
    ) -> None:
        """Refresh the active setlist after edits, keeping track position."""
        if (
            self._context_kind != CONTEXT_SETLIST
            or self._context_id != setlist_id
        ):
            return
        new_songs = _validate_song_ids(song_ids)
        if not new_songs:
            self.clear_setlist()
            return

        if name is not None:
            self._context_name = str(name)

        old_songs = self._context_songs
        old_idx = self._context_index
        new_set = set(new_songs)

        if current_song_id and current_song_id in new_set:
            new_idx = new_songs.index(current_song_id)
        elif 0 <= old_idx < len(old_songs) and old_songs[old_idx] in new_set:
            new_idx = new_songs.index(old_songs[old_idx])
        else:
            # Find the nearest remaining song at or before the previous index so
            # advancing next continues right after where playback left off.
            new_idx = -1
            upper = min(old_idx - 1, len(old_songs) - 1)
            for idx in range(upper, -1, -1):
                candidate = old_songs[idx]
                if candidate in new_set:
                    new_idx = new_songs.index(candidate)
                    break

        self._context_songs = new_songs
        self._context_index = new_idx
        self._shuffle_queue = [
            sid for sid in self._shuffle_queue if sid in new_set
        ]
        self._shuffle_visited &= new_set

    def clear_setlist(self) -> None:
        """Return playback context to the library collection."""
        self._context_kind = CONTEXT_LIBRARY
        self._context_id = None
        self._context_name = ""
        self._context_songs = []
        self._context_index = -1
        self._shuffle_queue.clear()
        self._shuffle_visited.clear()

    def upcoming_context_songs(self) -> list[str]:
        """Return remaining setlist tracks after the current position."""
        if not self.is_setlist_active:
            return []
        start = max(0, self._context_index + 1)
        return list(self._context_songs[start:])

    # ------------------------------------------------------------------
    # History and maintenance
    # ------------------------------------------------------------------

    def record_history(self, song_id: str | None) -> None:
        """Push *song_id* to navigation history (avoiding back-to-back dupes)."""
        if not isinstance(song_id, str) or not song_id.strip():
            return
        if self._history and self._history[-1] == song_id:
            return
        self._history.append(song_id)
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

    def remove_song(self, song_id: str) -> None:
        """Remove all references to *song_id* across queue, history, and context."""
        if not isinstance(song_id, str) or not song_id:
            return
        self.prune([
            sid for sid in (
                set(self._up_next)
                | set(self._history)
                | set(self._context_songs)
                | set(self._shuffle_queue)
            )
            if sid != song_id
        ])

    def prune(self, valid_song_ids: Iterable[str]) -> None:
        """Drop every ID not in *valid_song_ids* while preserving indices."""
        valid = {sid for sid in valid_song_ids if isinstance(sid, str) and sid}
        self._up_next = [sid for sid in self._up_next if sid in valid]
        self._history = [sid for sid in self._history if sid in valid]
        self._shuffle_queue = [
            sid for sid in self._shuffle_queue if sid in valid
        ]
        self._shuffle_visited &= valid

        if self._context_kind == CONTEXT_SETLIST:
            if not self._context_songs:
                self.clear_setlist()
                return
            kept: list[str] = []
            removed_up_to_index = 0
            for idx, sid in enumerate(self._context_songs):
                if sid in valid:
                    kept.append(sid)
                elif idx <= self._context_index:
                    removed_up_to_index += 1
            if not kept:
                self.clear_setlist()
            else:
                self._context_songs = kept
                self._context_index = max(
                    -1,
                    min(len(kept) - 1, self._context_index - removed_up_to_index),
                )

    def reset_shuffle(self) -> None:
        """Clear pending shuffle candidates."""
        self._shuffle_queue.clear()
        self._shuffle_visited.clear()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def advance_next(
        self,
        current_song_id: str | None,
        library_song_ids: Sequence[str],
        *,
        repeat_mode: str = REPEAT_OFF,
        shuffle_enabled: bool = False,
        from_auto_finish: bool = False,
        is_playable: Callable[[str], bool] | None = None,
    ) -> str | None:
        """Resolve and consume the next song ID to play."""
        if repeat_mode not in _VALID_REPEAT_MODES:
            repeat_mode = REPEAT_OFF

        def playable(sid: str) -> bool:
            return True if is_playable is None else bool(is_playable(sid))

        # 1. Explicit Up Next queue always takes precedence over context.
        while self._up_next:
            candidate = self._up_next.pop(0)
            if playable(candidate):
                self.record_history(current_song_id)
                return candidate

        # 2. Library auto-finish stops unless Repeat All or active setlist.
        if (
            from_auto_finish
            and repeat_mode == REPEAT_OFF
            and not self.is_setlist_active
        ):
            return None

        if self.is_setlist_active:
            return self._advance_setlist_next(
                current_song_id,
                repeat_mode=repeat_mode,
                shuffle_enabled=shuffle_enabled,
                from_auto_finish=from_auto_finish,
                playable=playable,
            )

        return self._advance_library_next(
            current_song_id,
            library_song_ids,
            repeat_mode=repeat_mode,
            shuffle_enabled=shuffle_enabled,
            playable=playable,
        )

    def _advance_setlist_next(
        self,
        current_song_id: str | None,
        *,
        repeat_mode: str,
        shuffle_enabled: bool,
        from_auto_finish: bool,
        playable: Callable[[str], bool],
    ) -> str | None:
        songs = self._context_songs
        if not songs:
            return None

        if shuffle_enabled:
            if current_song_id and current_song_id in songs:
                self._shuffle_visited.add(current_song_id)
            song_set = set(songs)
            self._shuffle_queue = [
                sid
                for sid in self._shuffle_queue
                if sid in song_set and sid != current_song_id and playable(sid)
            ]
            if not self._shuffle_queue:
                unvisited = [
                    sid
                    for sid in songs
                    if sid not in self._shuffle_visited
                    and sid != current_song_id
                    and playable(sid)
                ]
                if unvisited:
                    candidates = list(unvisited)
                else:
                    if from_auto_finish and repeat_mode == REPEAT_OFF:
                        return None
                    candidates = [
                        sid
                        for sid in songs
                        if sid != current_song_id and playable(sid)
                    ]
                    if not candidates:
                        return None
                    self._shuffle_visited = (
                        {current_song_id}
                        if current_song_id in song_set
                        else set()
                    )
                random.shuffle(candidates)
                self._shuffle_queue = candidates

            nxt = self._shuffle_queue.pop(0)
            self._shuffle_visited.add(nxt)
            self._context_index = songs.index(nxt)
            self.record_history(current_song_id)
            return nxt

        n = len(songs)
        for step in range(1, n + 1):
            raw_idx = self._context_index + step
            if raw_idx >= n:
                if repeat_mode != REPEAT_ALL:
                    return None
                idx = raw_idx % n
            else:
                idx = raw_idx
            candidate = songs[idx]
            if playable(candidate):
                self._context_index = idx
                self.record_history(current_song_id)
                return candidate
        return None

    def _advance_library_next(
        self,
        current_song_id: str | None,
        library_song_ids: Sequence[str],
        *,
        repeat_mode: str,
        shuffle_enabled: bool,
        playable: Callable[[str], bool],
    ) -> str | None:
        songs = [sid for sid in library_song_ids if isinstance(sid, str) and sid]
        if not songs:
            return None

        if shuffle_enabled:
            song_set = set(songs)
            self._shuffle_queue = [
                sid
                for sid in self._shuffle_queue
                if sid in song_set and sid != current_song_id and playable(sid)
            ]
            if not self._shuffle_queue:
                candidates = [
                    sid
                    for sid in songs
                    if sid != current_song_id and playable(sid)
                ]
                if not candidates:
                    return None
                random.shuffle(candidates)
                self._shuffle_queue = candidates
            nxt = self._shuffle_queue.pop(0)
            self.record_history(current_song_id)
            return nxt

        n = len(songs)
        start_idx = (
            songs.index(current_song_id) if current_song_id in songs else -1
        )
        for step in range(1, n + 1):
            raw_idx = start_idx + step
            if raw_idx >= n:
                if repeat_mode != REPEAT_ALL:
                    return None
                idx = raw_idx % n
            else:
                idx = raw_idx
            candidate = songs[idx]
            if playable(candidate):
                self.record_history(current_song_id)
                return candidate
        return None

    def advance_previous(
        self,
        current_song_id: str | None,
        library_song_ids: Sequence[str],
        *,
        repeat_mode: str = REPEAT_OFF,
        is_playable: Callable[[str], bool] | None = None,
    ) -> str | None:
        """Resolve the previous song ID from history or context order."""
        if repeat_mode not in _VALID_REPEAT_MODES:
            repeat_mode = REPEAT_OFF

        def playable(sid: str) -> bool:
            return True if is_playable is None else bool(is_playable(sid))

        while self._history:
            prev_id = self._history.pop()
            if prev_id == current_song_id:
                continue
            if playable(prev_id):
                if self.is_setlist_active and prev_id in self._context_songs:
                    self._context_index = self._context_songs.index(prev_id)
                return prev_id

        if self.is_setlist_active:
            songs = self._context_songs
            n = len(songs)
            start_idx = (
                self._context_index if 0 <= self._context_index < n else 0
            )
            for step in range(1, n + 1):
                raw_idx = start_idx - step
                if raw_idx < 0:
                    if repeat_mode != REPEAT_ALL:
                        return None
                    idx = raw_idx % n
                else:
                    idx = raw_idx
                candidate = songs[idx]
                if playable(candidate):
                    self._context_index = idx
                    return candidate
            return None

        songs = [sid for sid in library_song_ids if isinstance(sid, str) and sid]
        n = len(songs)
        if not n:
            return None
        start_idx = (
            songs.index(current_song_id) if current_song_id in songs else 1
        )
        for step in range(1, n + 1):
            raw_idx = start_idx - step
            if raw_idx < 0:
                if repeat_mode != REPEAT_ALL:
                    return None
                idx = raw_idx % n
            else:
                idx = raw_idx
            candidate = songs[idx]
            if playable(candidate):
                return candidate
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize queue and active context into JSON-safe primitives."""
        return {
            "up_next": list(self._up_next),
            "context_kind": self._context_kind,
            "context_id": self._context_id,
            "context_name": self._context_name,
            "context_songs": list(self._context_songs),
            "context_index": self._context_index,
            "history": list(self._history),
        }

    @classmethod
    def from_dict(cls, data) -> PlayQueue:
        """Reconstruct a :class:`PlayQueue`, ignoring malformed fields."""
        queue = cls()
        if not isinstance(data, dict):
            return queue

        queue._up_next = _clean_id_list(data.get("up_next"))
        queue._history = _clean_id_list(data.get("history"))[-_MAX_HISTORY:]

        kind = data.get("context_kind")
        cid = data.get("context_id")
        csongs = _clean_id_list(data.get("context_songs"))
        if (
            kind == CONTEXT_SETLIST
            and isinstance(cid, str)
            and cid.strip()
            and csongs
        ):
            queue._context_kind = CONTEXT_SETLIST
            queue._context_id = cid
            cname = data.get("context_name")
            queue._context_name = cname if isinstance(cname, str) else ""
            queue._context_songs = csongs
            raw_idx = data.get("context_index")
            if (
                isinstance(raw_idx, int)
                and not isinstance(raw_idx, bool)
                and -1 <= raw_idx < len(csongs)
            ):
                queue._context_index = raw_idx
            else:
                queue._context_index = -1

        return queue


def _clean_id_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item for item in value
        if isinstance(item, str) and item.strip()
    ]
