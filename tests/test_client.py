"""Tests for the API client retry logic."""

import json
from unittest.mock import MagicMock, patch

import pytest

from eval_harness.client import EvalClient
from eval_harness.config import EvalConfig


@pytest.fixture
def config():
    cfg = EvalConfig.default()
    cfg.retry.max_retries = 2
    cfg.retry.base_delay = 0.01  # Fast tests
    cfg.retry.max_delay = 0.1
    cfg.server.timeout_seconds = 5
    return cfg


@pytest.fixture
def client(config):
    c = EvalClient(config)
    # Disable throttle delay for fast tests
    c._throttle._min_interval = 0.0
    return c


class TestRetryLogic:
    """Test client retry behavior."""

    @patch("eval_harness.client.requests.Session.post")
    def test_successful_request(self, mock_post, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        mock_post.return_value = mock_resp

        result = client.chat_completion([{"role": "user", "content": "Hi"}])
        assert result.content == "Hello"
        assert result.error_category == "none"

    @patch("eval_harness.client.requests.Session.post")
    def test_rate_limit_retry(self, mock_post, client):
        # First call returns 429, second returns 200
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "0.01"}
        mock_429.text = "Rate limited"

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }

        mock_post.side_effect = [mock_429, mock_200]

        result = client.chat_completion([{"role": "user", "content": "Hi"}])
        assert result.content == "OK"
        assert result.retries_used == 1

    @patch("eval_harness.client.requests.Session.post")
    def test_max_retries_exhausted(self, mock_post, client):
        mock_503 = MagicMock()
        mock_503.status_code = 503
        mock_503.headers = {}
        mock_503.text = "Server error"

        mock_post.return_value = mock_503

        result = client.chat_completion([{"role": "user", "content": "Hi"}])
        assert result.error_category == "server_error"
        assert result.content == ""

    @patch("eval_harness.client.requests.Session.post")
    def test_truncated_json_retry(self, mock_post, client):
        # First call returns truncated JSON, second returns valid
        mock_truncated = MagicMock()
        mock_truncated.status_code = 200
        mock_truncated.json.side_effect = json.JSONDecodeError("Truncated", "", 0)
        mock_truncated.text = '{"choices": [{"mess'

        mock_valid = MagicMock()
        mock_valid.status_code = 200
        mock_valid.json.return_value = {
            "choices": [{"message": {"content": "Fixed"}}]
        }

        mock_post.side_effect = [mock_truncated, mock_valid]

        result = client.chat_completion([{"role": "user", "content": "Hi"}])
        assert result.content == "Fixed"

    def test_error_logging(self, client):
        assert len(client.error_log) == 0
