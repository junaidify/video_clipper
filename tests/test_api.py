"""Tests for Flask app API routes."""
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

@pytest.fixture
def client():
    """Flask test client with heavy dependencies mocked."""
    with patch("media.library.VideoLibrary"), \
         patch("training.trainer.PatternTrainer"):
        from app import app as flask_app
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            yield c


class TestLandingPage:
    def test_get_landing(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_get_app(self, client):
        resp = client.get("/app")
        assert resp.status_code == 200


class TestSettingsAPI:
    def test_get_settings(self, client):
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "ffmpeg_installed" in data
        assert "llm_available" in data
        assert "is_local" in data
        assert "cookie_auth" in data


class TestPlatformsAPI:
    def test_get_platforms(self, client):
        resp = client.get("/api/platforms")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fully_supported" in data
        assert "not_supported" in data
        assert "total_platforms" in data


class TestCheckUrlAPI:
    def test_missing_url(self, client):
        resp = client.post("/api/check-url",
                           data=json.dumps({"url": ""}),
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is False

    def test_invalid_url(self, client):
        resp = client.post("/api/check-url",
                           data=json.dumps({"url": "not-a-url"}),
                           content_type="application/json")
        data = resp.get_json()
        assert data["valid"] is False

    def test_drm_url_rejected(self, client):
        resp = client.post("/api/check-url",
                           data=json.dumps({"url": "https://netflix.com/watch/123"}),
                           content_type="application/json")
        data = resp.get_json()
        assert data["valid"] is False
        assert "DRM" in data["error"]

    @patch("app.get_video_info")
    def test_valid_url_with_mock(self, mock_info, client):
        mock_info.return_value = {
            "title": "Test Video", "duration": 120,
            "platform": "YouTube", "url": "https://youtube.com/watch?v=abc"
        }
        resp = client.post("/api/check-url",
                           data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                           content_type="application/json")
        data = resp.get_json()
        assert data["valid"] is True
        assert data["info"]["title"] == "Test Video"


class TestJobStatus:
    def test_nonexistent_job(self, client):
        resp = client.get("/api/status/nonexistent-job-id-12345")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data


class TestEditorPage:
    def test_editor_without_params(self, client):
        resp = client.get("/editor")
        # Should either render or redirect, both are valid
        assert resp.status_code in (200, 302, 400)
