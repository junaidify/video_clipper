"""Tests for video_clipper.media_management.downloader module (no network calls)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from video_clipper.media_management.downloader import (
    is_valid_url, is_drm_platform, _extract_domain, _detect_platform,
    DownloadResult,
)


class TestIsValidUrl:
    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=abc",
        "http://example.com/video.mp4",
        "https://tiktok.com/@user/video/123",
    ])
    def test_valid_urls(self, url):
        assert is_valid_url(url) is True

    @pytest.mark.parametrize("url", [
        "not-a-url",
        "ftp://files.example.com/video.mp4",
        "youtube.com/watch?v=abc",
        "",
        "   ",
    ])
    def test_invalid_urls(self, url):
        assert is_valid_url(url) is False

    def test_strips_whitespace(self):
        assert is_valid_url("  https://example.com  ") is True


class TestIsDrmPlatform:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.netflix.com/watch/12345", "Netflix"),
        ("https://www.disneyplus.com/video/abc", "Disney+"),
        ("https://www.hulu.com/watch/xyz", "Hulu"),
        ("https://max.com/movie/123", "Max"),
    ])
    def test_drm_platforms_detected(self, url, expected):
        result = is_drm_platform(url)
        assert result is not None
        assert expected.lower() in result.lower()

    def test_youtube_not_drm(self):
        assert is_drm_platform("https://youtube.com/watch?v=abc") is None

    def test_tiktok_not_drm(self):
        assert is_drm_platform("https://tiktok.com/@user/video/123") is None

    def test_amazon_shopping_not_drm(self):
        assert is_drm_platform("https://amazon.com/product/abc") is None

    def test_amazon_prime_video_is_drm(self):
        result = is_drm_platform("https://amazon.com/gp/video/detail/abc")
        assert result is not None


class TestExtractDomain:
    def test_basic(self):
        assert _extract_domain("https://www.youtube.com/watch") == "youtube.com"

    def test_strips_www(self):
        assert _extract_domain("https://www.example.com") == "example.com"

    def test_strips_mobile(self):
        assert _extract_domain("https://m.youtube.com/watch") == "youtube.com"

    def test_no_scheme(self):
        assert _extract_domain("youtube.com/watch") == "youtube.com"

    def test_empty_string(self):
        assert _extract_domain("") == ""


class TestDetectPlatform:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.youtube.com/watch?v=abc", "YouTube"),
        ("https://youtu.be/abc", "YouTube"),
        ("https://www.tiktok.com/@user/video/123", "TikTok"),
        ("https://www.instagram.com/reel/abc", "Instagram"),
        ("https://twitter.com/user/status/123", "X/Twitter"),
        ("https://x.com/user/status/123", "X/Twitter"),
        ("https://vimeo.com/12345", "Vimeo"),
        ("https://reddit.com/r/test/comments/abc", "Reddit"),
    ])
    def test_known_platforms(self, url, expected):
        assert _detect_platform(url) == expected

    def test_unknown_platform(self):
        result = _detect_platform("https://obscure-site.org/video")
        assert isinstance(result, str)
        assert len(result) > 0


class TestDownloadResult:
    def test_success_result(self):
        r = DownloadResult(success=True, file_path="/tmp/video.mp4",
                           title="Test", duration=120.0, platform="YouTube")
        assert r.success is True
        assert r.error is None

    def test_failure_result(self):
        r = DownloadResult(success=False, error="Network error", platform="TikTok")
        assert r.success is False
        assert r.file_path is None
