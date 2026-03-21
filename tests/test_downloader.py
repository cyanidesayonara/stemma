"""Tests for the YouTube downloader module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.downloader import (
    is_supported_url,
    extract_metadata,
    download_audio,
    DownloadError,
)


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

        result = download_audio(
            "https://youtu.be/dQw4w9WgXcQ", output_path
        )
        assert os.path.isdir(str(tmp_path / "subdir"))

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_download_error_raises(self, mock_ydl_class, tmp_path):
        output_path = str(tmp_path / "audio.mp3")

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.download.side_effect = Exception("403 Forbidden")

        with pytest.raises(DownloadError, match="403 Forbidden"):
            download_audio("https://youtu.be/abc123", output_path)

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
