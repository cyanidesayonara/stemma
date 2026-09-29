"""Tests for the YouTube downloader module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

import src.downloader as downloader
from src.downloader import (
    is_supported_url,
    extract_metadata,
    download_audio,
    check_ffmpeg,
    DownloadError,
    is_transient_error,
)
from src.import_messages import MSG_HTTP, describe_error, strip_ansi

# The real yt-dlp message from the failed first import (colour codes and
# all): YouTube refused the first media URL, and Retry worked.
RAW_403 = (
    "\x1b[0;31mERROR:\x1b[0m unable to download video data: "
    "HTTP Error 403: Forbidden"
)


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Record backoff waits instead of sleeping through them."""
    waits = []
    monkeypatch.setattr(downloader, "_sleep", waits.append)
    return waits


def _ydl_sequence(mock_ydl_class, output_path, outcomes):
    """Make each YoutubeDL instance run the next outcome in *outcomes*.

    An outcome is an exception to raise from download(), or None to
    succeed by writing the output file. Returns the list of option dicts
    each attempt was built with.
    """
    remaining = list(outcomes)
    built = []

    def make(opts):
        built.append(opts)
        outcome = remaining.pop(0)
        ydl = MagicMock()

        def fake_download(urls):
            if outcome is not None:
                raise outcome
            with open(output_path, "wb") as f:
                f.write(b"fake mp3 data")

        ydl.download.side_effect = fake_download
        context = MagicMock()
        context.__enter__ = MagicMock(return_value=ydl)
        context.__exit__ = MagicMock(return_value=False)
        return context

    mock_ydl_class.side_effect = make
    return built


class TestURLValidation:
    """Test URL pattern matching."""

    def test_youtube_watch_url(self):
        assert is_supported_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_short_url(self):
        assert is_supported_url("https://youtu.be/dQw4w9WgXcQ")

    def test_youtube_music_url(self):
        assert is_supported_url("https://music.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_no_scheme(self):
        assert is_supported_url("youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_shorts_url(self):
        assert is_supported_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")

    def test_youtube_mobile_url(self):
        assert is_supported_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_live_url(self):
        assert is_supported_url("https://www.youtube.com/live/dQw4w9WgXcQ")

    def test_youtube_www_music_url(self):
        assert is_supported_url(
            "https://www.music.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    def test_empty_string(self):
        assert not is_supported_url("")

    def test_random_text(self):
        assert not is_supported_url("not a url at all")

    def test_local_file_path(self):
        assert not is_supported_url("C:\\Users\\music\\song.mp3")

    def test_other_website(self):
        assert not is_supported_url("https://www.google.com")

    def test_embedded_url_in_text(self):
        """Text containing a YouTube URL but not starting with one."""
        assert not is_supported_url("DO NOT VISIT youtube.com/watch?v=malware")

    def test_fake_domain_prefix(self):
        """Domain that ends with youtube.com but isn't."""
        assert not is_supported_url("https://fakeyoutube.com/watch?v=abc")

    def test_youtube_playlist_url(self):
        assert is_supported_url(
            "https://www.youtube.com/watch?v=abc123&list=PLxyz"
        )


class TestExtractMetadata:
    """Test metadata extraction from URLs."""

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_returns_title_and_artist(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "title": "Never Gonna Give You Up",
            "uploader": "Rick Astley",
            "artist": "Rick Astley",
        }

        title, artist = extract_metadata("https://youtu.be/dQw4w9WgXcQ")
        assert title == "Never Gonna Give You Up"
        assert artist == "Rick Astley"

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_falls_back_to_uploader(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "title": "Some Video",
            "uploader": "SomeChannel",
        }

        title, artist = extract_metadata("https://youtu.be/abc123")
        assert title == "Some Video"
        assert artist == "SomeChannel"

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_missing_metadata_uses_defaults(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {}

        title, artist = extract_metadata("https://youtu.be/abc123")
        assert title == "Untitled"
        assert artist == "Unknown Artist"

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_error_raises_download_error(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = Exception("network error")

        with pytest.raises(DownloadError, match="network error"):
            extract_metadata("https://youtu.be/abc123")

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_extract_info_returns_none(self, mock_ydl_class):
        """Private/deleted videos can cause extract_info to return None."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = None

        with pytest.raises(DownloadError):
            extract_metadata("https://youtu.be/private123")


class TestCheckFFmpeg:
    """Test ffmpeg availability check."""

    @patch("src.downloader.imageio_ffmpeg.get_ffmpeg_exe",
           return_value="/bundled/ffmpeg")
    def test_available_via_imageio(self, mock_get):
        """imageio-ffmpeg bundled binary satisfies the check."""
        assert check_ffmpeg() is True

    @patch("src.downloader.imageio_ffmpeg.get_ffmpeg_exe",
           side_effect=RuntimeError("no binary"))
    @patch("src.downloader.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_available_via_path_fallback(self, mock_which, mock_get):
        """Falls back to PATH when imageio-ffmpeg raises."""
        assert check_ffmpeg() is True

    @patch("src.downloader.imageio_ffmpeg.get_ffmpeg_exe",
           side_effect=RuntimeError("no binary"))
    @patch("src.downloader.shutil.which", return_value=None)
    def test_missing_when_both_unavailable(self, mock_which, mock_get):
        """Returns False only when neither source has ffmpeg."""
        assert check_ffmpeg() is False


class TestDownloadAudio:
    """Test audio download."""

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_creates_file(self, mock_ydl_class, tmp_path):
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        # Simulate yt-dlp creating the output file
        def fake_download(urls):
            with open(output_path, "wb") as f:
                f.write(b"fake mp3 data")

        mock_ydl.download.side_effect = fake_download

        result = download_audio(
            "https://youtu.be/dQw4w9WgXcQ", output_path
        )
        assert result == output_path
        assert os.path.isfile(output_path)

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_creates_parent_dir(self, mock_ydl_class, tmp_path):
        output_path = str(tmp_path / "subdir" / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        def fake_download(urls):
            with open(output_path, "wb") as f:
                f.write(b"fake mp3 data")

        mock_ydl.download.side_effect = fake_download

        download_audio("https://youtu.be/dQw4w9WgXcQ", output_path)
        assert os.path.isdir(str(tmp_path / "subdir"))

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_error_raises(self, mock_ydl_class, tmp_path):
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.download.side_effect = Exception("HTTP Error 403: Forbidden")

        with pytest.raises(DownloadError, match="403: Forbidden"):
            download_audio("https://youtu.be/abc123", output_path)
        # A persistent 403 is tried a bounded number of times, then raised.
        assert mock_ydl_class.call_count == downloader._MAX_ATTEMPTS

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_passes_correct_options(self, mock_ydl_class, tmp_path):
        """Verify yt-dlp is configured for audio-only MP3 extraction."""
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        # yt-dlp creates the file at stem + .mp3 (from postprocessor)
        def fake_download(urls):
            with open(output_path, "wb") as f:
                f.write(b"fake")

        mock_ydl.download.side_effect = fake_download

        download_audio("https://youtu.be/dQw4w9WgXcQ", output_path)

        # outtmpl should be the stem WITHOUT extension, because
        # FFmpegExtractAudio appends the codec extension itself.
        opts = mock_ydl_class.call_args[0][0]
        assert opts["format"] == "bestaudio/best"
        stem = os.path.splitext(output_path)[0]
        assert opts["outtmpl"] == stem

    @patch("src.downloader.imageio_ffmpeg.get_ffmpeg_exe",
           return_value="/bundled/ffmpeg")
    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_ffmpeg_location_passed_to_ytdlp(self, mock_ydl_class, mock_get,
                                              tmp_path):
        """download_audio passes ffmpeg_location so yt-dlp finds ffmpeg."""
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        def fake_download(urls):
            with open(output_path, "wb") as f:
                f.write(b"fake")

        mock_ydl.download.side_effect = fake_download

        download_audio("https://youtu.be/dQw4w9WgXcQ", output_path)

        opts = mock_ydl_class.call_args[0][0]
        assert opts["ffmpeg_location"] == "/bundled/ffmpeg"

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_progress_callback_invoked(self, mock_ydl_class, tmp_path):
        """Verify progress hook is wired up when callback is provided."""
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        def fake_download(urls):
            with open(output_path, "wb") as f:
                f.write(b"fake")

        mock_ydl.download.side_effect = fake_download

        callback = MagicMock()
        download_audio(
            "https://youtu.be/dQw4w9WgXcQ", output_path,
            progress_callback=callback,
        )

        # Should have a progress_hooks entry in opts
        opts = mock_ydl_class.call_args[0][0]
        assert len(opts["progress_hooks"]) == 1

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_noplaylist_option_set(self, mock_ydl_class, tmp_path):
        """Playlist URLs should only download a single video."""
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        def fake_download(urls):
            with open(output_path, "wb") as f:
                f.write(b"fake")

        mock_ydl.download.side_effect = fake_download

        download_audio(
            "https://www.youtube.com/watch?v=abc123&list=PLxyz",
            output_path,
        )

        opts = mock_ydl_class.call_args[0][0]
        assert opts["noplaylist"] is True

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_raises_when_output_file_missing(self, mock_ydl_class, tmp_path):
        """Raise DownloadError if yt-dlp exits without creating the file."""
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        # yt-dlp "succeeds" but does not create the output file.
        mock_ydl.download.side_effect = lambda urls: None

        with pytest.raises(DownloadError, match="not found"):
            download_audio("https://youtu.be/abc123", output_path)


class TestExtractMetadataNoPlaylist:
    """Test that extract_metadata also sets noplaylist."""

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_noplaylist_in_metadata_opts(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"title": "T", "uploader": "U"}

        extract_metadata("https://youtu.be/abc123")

        opts = mock_ydl_class.call_args[0][0]
        assert opts["noplaylist"] is True


class TestTransientErrors:
    """Which failures are worth a fresh extraction."""

    @pytest.mark.parametrize("message", [
        RAW_403,
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
        "HTTP Error 503: Service Unavailable",
        "HTTP Error 502: Bad Gateway",
        "ERROR: [download] Got error: The read operation timed out",
        "Connection reset by peer",
        "IncompleteRead(1024 bytes read, 2048 more expected)",
        "Remote end closed connection without response",
    ])
    def test_transient(self, message):
        assert is_transient_error(message)

    @pytest.mark.parametrize("message", [
        "ERROR: [youtube] abc: Private video. Sign in if you've been granted "
        "access to this video",
        "ERROR: [youtube] abc: Video unavailable. This video has been "
        "removed by the uploader",
        "ERROR: Unsupported URL: https://example.com/",
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot",
        "ERROR: [youtube] abc: Sign in to confirm your age. HTTP Error 403",
        "ERROR: [youtube] abc: This video is not available in your country",
        "HTTP Error 404: Not Found",
        "HTTP Error 429: Too Many Requests",
        "ERROR: [youtube] abc: Requested format is not available",
        "Postprocessing: ffprobe and ffmpeg not found",
        "",
    ])
    def test_permanent(self, message):
        assert not is_transient_error(message)


class TestDownloadRetry:
    """A refused first media URL is retried with a fresh extraction."""

    URL = "https://youtu.be/dQw4w9WgXcQ"

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_403_then_success_is_retried(self, mock_ydl_class, tmp_path,
                                         no_backoff_sleep):
        output_path = str(tmp_path / "audio.mp3")
        built = _ydl_sequence(mock_ydl_class, output_path, [
            yt_dlp.utils.DownloadError(RAW_403), None,
        ])

        assert download_audio(self.URL, output_path) == output_path
        # Each attempt builds a new YoutubeDL, so the media URL is
        # extracted afresh rather than the refused one reused.
        assert len(built) == 2
        assert sum(no_backoff_sleep) > 0

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_backoff_grows(self, mock_ydl_class, tmp_path, no_backoff_sleep):
        output_path = str(tmp_path / "audio.mp3")
        err = yt_dlp.utils.DownloadError(RAW_403)
        _ydl_sequence(mock_ydl_class, output_path, [err, err, None])

        download_audio(self.URL, output_path)
        first, second = downloader._BACKOFF_S[:2]
        assert 0 < first < second
        assert sum(no_backoff_sleep) == pytest.approx(first + second)

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_permanent_failure_is_not_retried(self, mock_ydl_class, tmp_path,
                                              no_backoff_sleep):
        output_path = str(tmp_path / "audio.mp3")
        built = _ydl_sequence(mock_ydl_class, output_path, [
            yt_dlp.utils.DownloadError(
                "\x1b[0;31mERROR:\x1b[0m [youtube] abc: Private video"
            ),
        ])

        with pytest.raises(DownloadError, match="Private video"):
            download_audio(self.URL, output_path)
        assert len(built) == 1
        assert no_backoff_sleep == []

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_gives_up_after_the_last_attempt(self, mock_ydl_class, tmp_path):
        output_path = str(tmp_path / "audio.mp3")
        err = yt_dlp.utils.DownloadError(RAW_403)
        built = _ydl_sequence(
            mock_ydl_class, output_path, [err] * downloader._MAX_ATTEMPTS,
        )

        with pytest.raises(DownloadError, match="403"):
            download_audio(self.URL, output_path)
        assert len(built) == downloader._MAX_ATTEMPTS
        assert 2 <= downloader._MAX_ATTEMPTS <= 3

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_progress_hook_kept_on_every_attempt(self, mock_ydl_class,
                                                 tmp_path):
        output_path = str(tmp_path / "audio.mp3")
        built = _ydl_sequence(mock_ydl_class, output_path, [
            yt_dlp.utils.DownloadError(RAW_403), None,
        ])
        callback = MagicMock()

        download_audio(self.URL, output_path, progress_callback=callback)
        assert [opts["progress_hooks"] for opts in built] == [
            [callback], [callback],
        ]

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_cancel_before_retry_stops(self, mock_ydl_class, tmp_path):
        output_path = str(tmp_path / "audio.mp3")
        built = _ydl_sequence(mock_ydl_class, output_path, [
            yt_dlp.utils.DownloadError(RAW_403), None,
        ])

        with pytest.raises(DownloadError, match="cancelled"):
            download_audio(self.URL, output_path, should_cancel=lambda: True)
        assert len(built) == 1

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_cancel_during_backoff_stops(self, mock_ydl_class, tmp_path,
                                         no_backoff_sleep):
        output_path = str(tmp_path / "audio.mp3")
        built = _ydl_sequence(mock_ydl_class, output_path, [
            yt_dlp.utils.DownloadError(RAW_403), None,
        ])

        def cancelled() -> bool:
            # Cancel arrives once the backoff has started waiting.
            return bool(no_backoff_sleep)

        with pytest.raises(DownloadError, match="cancelled"):
            download_audio(self.URL, output_path, should_cancel=cancelled)
        assert len(built) == 1
        assert sum(no_backoff_sleep) < downloader._BACKOFF_S[0]


class TestNoColourCodes:
    """yt-dlp messages reach the log and the UI without ANSI codes."""

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_asks_ytdlp_for_no_colour(self, mock_ydl_class,
                                               tmp_path):
        output_path = str(tmp_path / "audio.mp3")
        built = _ydl_sequence(mock_ydl_class, output_path, [None])

        download_audio("https://youtu.be/abc123", output_path)
        assert built[0]["no_color"] is True

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_metadata_asks_ytdlp_for_no_colour(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"title": "T", "uploader": "U"}

        extract_metadata("https://youtu.be/abc123")
        assert mock_ydl_class.call_args[0][0]["no_color"] is True

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_error_text_is_stripped(self, mock_ydl_class, tmp_path):
        output_path = str(tmp_path / "audio.mp3")
        err = yt_dlp.utils.DownloadError(RAW_403)
        _ydl_sequence(
            mock_ydl_class, output_path, [err] * downloader._MAX_ATTEMPTS,
        )

        with pytest.raises(DownloadError) as caught:
            download_audio("https://youtu.be/abc123", output_path)
        assert "\x1b" not in str(caught.value)
        assert str(caught.value).startswith("ERROR: unable to download")
        assert describe_error(caught.value) == MSG_HTTP

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_metadata_error_text_is_stripped(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError(
            "\x1b[0;31mERROR:\x1b[0m [youtube] abc: Private video"
        )

        with pytest.raises(DownloadError) as caught:
            extract_metadata("https://youtu.be/abc123")
        assert "\x1b" not in str(caught.value)

    def test_strip_ansi(self):
        assert strip_ansi(RAW_403) == (
            "ERROR: unable to download video data: HTTP Error 403: Forbidden"
        )
        assert strip_ansi("plain text") == "plain text"
        assert strip_ansi("\x1b[1m\x1b[33mwarn\x1b[0m") == "warn"

    def test_describe_error_never_shows_colour_codes(self):
        # A raw message with codes that nothing maps is shown as-is, so
        # it must be stripped there too.
        shown = describe_error("\x1b[0;31mERROR:\x1b[0m something odd")
        assert "\x1b" not in shown
        assert shown == "ERROR: something odd"

    def test_describe_error_logs_without_colour_codes(self, caplog):
        with caplog.at_level("WARNING"):
            describe_error("\x1b[0;31mERROR:\x1b[0m something odd", "ctx")
        assert "\x1b" not in caplog.text
