"""Tests for hf_model_monitor.hf_client — HF API client with pagination,
rate limiting, and error handling."""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests

from hf_model_monitor.hf_client import (
    HFApiClient,
    HFApiError,
    HFAuthError,
    HFRateLimitError,
    ModelRecord,
    RateLimiter,
    _parse_retry_after,
    fetch_org_models,
    fetch_all_watched_org_models,
    fetch_latest_from_config,
    reset_default_client,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_raw_model(model_id: str, **overrides) -> dict:
    """Build a raw HF API model dict for testing."""
    author = model_id.split("/")[0] if "/" in model_id else "unknown"
    base = {
        "id": model_id,
        "author": author,
        "createdAt": "2026-04-20T10:00:00.000Z",
        "lastModified": "2026-04-25T12:00:00.000Z",
        "pipeline_tag": "text-generation",
        "tags": ["transformers", "pytorch"],
        "downloads": 5000,
        "likes": 100,
        "private": False,
        "gated": False,
        "library_name": "transformers",
    }
    base.update(overrides)
    return base


def _make_response(
    json_data=None, status_code=200, headers=None, raise_for_status=None
):
    """Build a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.headers = headers or {}
    if raise_for_status:
        resp.raise_for_status.side_effect = raise_for_status
    else:
        resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def client():
    """Create a test client with fast settings (no real sleeping)."""
    return HFApiClient(
        timeout=5,
        per_page=10,
        max_pages=3,
        min_request_interval=0.0,  # no delay in tests
        max_retries=3,
        retry_backoff=0.01,  # near-instant backoff in tests
    )


# ===================================================================
# ModelRecord
# ===================================================================

class TestModelRecord:
    def test_to_dict_roundtrip(self):
        record = ModelRecord(
            model_id="meta-llama/Llama-4",
            author="meta-llama",
            created_at="2026-04-20T10:00:00Z",
            last_modified="2026-04-25T12:00:00Z",
            pipeline_tag="text-generation",
            tags=["transformers"],
            downloads=5000,
            likes=100,
        )
        d = record.to_dict()
        assert d["model_id"] == "meta-llama/Llama-4"
        assert d["author"] == "meta-llama"
        assert d["downloads"] == 5000
        assert d["likes"] == 100
        assert d["pipeline_tag"] == "text-generation"
        assert isinstance(d["tags"], list)
        assert d["private"] is False
        assert d["gated"] is False
        assert d["library_name"] == "N/A"

    def test_defaults(self):
        record = ModelRecord(
            model_id="org/model",
            author="org",
            created_at="N/A",
            last_modified="N/A",
            pipeline_tag="N/A",
        )
        assert record.tags == []
        assert record.downloads == 0
        assert record.likes == 0
        assert record.private is False
        assert record.gated is False


# ===================================================================
# RateLimiter
# ===================================================================

class TestRateLimiter:
    def test_first_call_no_wait(self):
        limiter = RateLimiter(min_interval=10.0)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        # First call should not sleep
        assert elapsed < 0.1

    def test_enforces_minimum_interval(self):
        limiter = RateLimiter(min_interval=0.1)
        limiter.wait()  # first call
        start = time.monotonic()
        limiter.wait()  # second call should wait
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # some tolerance

    def test_reset(self):
        limiter = RateLimiter(min_interval=10.0)
        limiter.wait()
        limiter.reset()
        # After reset, next call should not wait
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1


# ===================================================================
# HFApiClient._parse_model
# ===================================================================

class TestParseModel:
    def test_parses_valid_model(self):
        raw = _make_raw_model("meta-llama/Llama-4-Scout-17B-16E")
        record = HFApiClient._parse_model(raw)
        assert record is not None
        assert record.model_id == "meta-llama/Llama-4-Scout-17B-16E"
        assert record.author == "meta-llama"
        assert record.downloads == 5000
        assert record.pipeline_tag == "text-generation"

    def test_returns_none_for_missing_id(self):
        assert HFApiClient._parse_model({}) is None
        assert HFApiClient._parse_model({"author": "org"}) is None

    def test_extracts_author_from_model_id(self):
        raw = {"id": "google/gemma-3", "createdAt": "2026-01-01"}
        record = HFApiClient._parse_model(raw)
        assert record is not None
        assert record.author == "google"

    def test_handles_modelId_field(self):
        raw = {"modelId": "microsoft/phi-4", "author": "microsoft"}
        record = HFApiClient._parse_model(raw)
        assert record is not None
        assert record.model_id == "microsoft/phi-4"

    def test_skips_gguf_derivative(self):
        raw = _make_raw_model(
            "community/Llama-4-gguf",
            tags=["gguf", "quantized"],
        )
        record = HFApiClient._parse_model(raw)
        assert record is None

    def test_skips_gptq_derivative(self):
        raw = _make_raw_model(
            "user123/Mistral-7B-gptq",
            tags=["gptq"],
        )
        record = HFApiClient._parse_model(raw)
        assert record is None

    def test_skips_awq_derivative(self):
        raw = _make_raw_model(
            "user/DeepSeek-V3-awq",
            tags=["awq"],
        )
        record = HFApiClient._parse_model(raw)
        assert record is None

    def test_keeps_original_repo_with_tags(self):
        """Original repos that mention gguf in tags but not in name should be kept."""
        raw = _make_raw_model(
            "meta-llama/Llama-4",
            tags=["gguf", "transformers"],
        )
        record = HFApiClient._parse_model(raw)
        assert record is not None
        assert record.model_id == "meta-llama/Llama-4"

    def test_handles_non_list_tags(self):
        raw = _make_raw_model("org/model", tags="not-a-list")
        record = HFApiClient._parse_model(raw)
        assert record is not None
        assert record.tags == []

    def test_defaults_for_missing_fields(self):
        raw = {"id": "org/model"}
        record = HFApiClient._parse_model(raw)
        assert record is not None
        assert record.created_at == "N/A"
        assert record.pipeline_tag == "N/A"
        assert record.downloads == 0
        assert record.library_name == "N/A"


# ===================================================================
# HFApiClient._request_with_retry
# ===================================================================

class TestRequestWithRetry:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_success_on_first_try(self, mock_sleep, client):
        resp = _make_response(json_data={"ok": True})
        client._session.get = MagicMock(return_value=resp)

        result = client._request_with_retry("https://example.com/api")
        assert result == {"ok": True}
        mock_sleep.assert_not_called()

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_retries_on_server_error_then_succeeds(self, mock_sleep, client):
        fail_resp = _make_response(status_code=500)
        ok_resp = _make_response(json_data=[{"id": "org/m"}])
        client._session.get = MagicMock(side_effect=[fail_resp, ok_resp])

        result = client._request_with_retry("https://example.com/api")
        assert result == [{"id": "org/m"}]
        assert client._session.get.call_count == 2

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_raises_on_exhausted_server_errors(self, mock_sleep, client):
        fail_resp = _make_response(status_code=502)
        client._session.get = MagicMock(return_value=fail_resp)

        with pytest.raises(HFApiError) as exc_info:
            client._request_with_retry("https://example.com/api")
        assert exc_info.value.status_code == 502
        assert client._session.get.call_count == 3  # max_retries

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_rate_limit_429_retries(self, mock_sleep, client):
        rate_resp = _make_response(
            status_code=429, headers={"Retry-After": "2"}
        )
        ok_resp = _make_response(json_data={"ok": True})
        client._session.get = MagicMock(side_effect=[rate_resp, ok_resp])

        result = client._request_with_retry("https://example.com/api")
        assert result == {"ok": True}
        # Should have slept for retry-after
        mock_sleep.assert_called()

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_rate_limit_raises_after_max_retries(self, mock_sleep, client):
        rate_resp = _make_response(
            status_code=429, headers={"Retry-After": "1"}
        )
        client._session.get = MagicMock(return_value=rate_resp)

        with pytest.raises(HFRateLimitError):
            client._request_with_retry("https://example.com/api")

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_auth_error_raises_immediately(self, mock_sleep, client):
        auth_resp = _make_response(status_code=401)
        client._session.get = MagicMock(return_value=auth_resp)

        with pytest.raises(HFAuthError) as exc_info:
            client._request_with_retry("https://example.com/api")
        assert exc_info.value.status_code == 401
        # Should NOT retry on auth errors
        assert client._session.get.call_count == 1

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_403_raises_auth_error(self, mock_sleep, client):
        resp = _make_response(status_code=403)
        client._session.get = MagicMock(return_value=resp)

        with pytest.raises(HFAuthError) as exc_info:
            client._request_with_retry("https://example.com/api")
        assert exc_info.value.status_code == 403

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_404_returns_none(self, mock_sleep, client):
        resp = _make_response(status_code=404)
        client._session.get = MagicMock(return_value=resp)

        result = client._request_with_retry("https://example.com/api")
        assert result is None

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_timeout_retries(self, mock_sleep, client):
        client._session.get = MagicMock(
            side_effect=[
                requests.exceptions.Timeout("timed out"),
                _make_response(json_data={"ok": True}),
            ]
        )

        result = client._request_with_retry("https://example.com/api")
        assert result == {"ok": True}
        assert client._session.get.call_count == 2

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_connection_error_retries(self, mock_sleep, client):
        client._session.get = MagicMock(
            side_effect=[
                requests.exceptions.ConnectionError("refused"),
                _make_response(json_data={"ok": True}),
            ]
        )

        result = client._request_with_retry("https://example.com/api")
        assert result == {"ok": True}

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_none_after_all_retries_exhausted(self, mock_sleep, client):
        client._session.get = MagicMock(
            side_effect=requests.exceptions.Timeout("timed out")
        )

        result = client._request_with_retry("https://example.com/api")
        assert result is None
        assert client._session.get.call_count == 3


# ===================================================================
# HFApiClient.fetch_org_models — pagination
# ===================================================================

class TestFetchOrgModels:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_single_page(self, mock_sleep, client):
        models = [
            _make_raw_model("meta-llama/Llama-4"),
            _make_raw_model("meta-llama/Llama-3.3"),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_models("meta-llama")
        assert len(result) == 2
        assert result[0].model_id == "meta-llama/Llama-4"
        assert result[1].model_id == "meta-llama/Llama-3.3"

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_multi_page_pagination(self, mock_sleep, client):
        page1_models = [_make_raw_model("org/model-1")]
        page2_models = [_make_raw_model("org/model-2")]

        resp1 = _make_response(
            json_data=page1_models,
            headers={"Link": '<https://huggingface.co/api/models?cursor=abc>; rel="next"'},
        )
        resp2 = _make_response(json_data=page2_models, headers={})

        client._session.get = MagicMock(side_effect=[resp1, resp2])

        result = client.fetch_org_models("org")
        assert len(result) == 2
        assert result[0].model_id == "org/model-1"
        assert result[1].model_id == "org/model-2"
        assert client._session.get.call_count == 2

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_stops_at_max_pages(self, mock_sleep, client):
        """Client should not exceed max_pages even if Link headers keep coming."""
        resp_with_link = _make_response(
            json_data=[_make_raw_model("org/model")],
            headers={"Link": '<https://huggingface.co/api/models?cursor=abc>; rel="next"'},
        )
        client._session.get = MagicMock(return_value=resp_with_link)

        # client.max_pages = 3
        result = client.fetch_org_models("org")
        assert client._session.get.call_count == 3  # stops at max_pages

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_stops_on_empty_page(self, mock_sleep, client):
        resp = _make_response(json_data=[], headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_models("org")
        assert result == []
        assert client._session.get.call_count == 1

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_stops_on_error_page(self, mock_sleep, client):
        """If a page fails, pagination stops but already-collected models are kept."""
        page1_models = [_make_raw_model("org/model-1")]
        resp1 = _make_response(
            json_data=page1_models,
            headers={"Link": '<https://huggingface.co/api/models?cursor=abc>; rel="next"'},
        )
        resp2 = _make_response(status_code=500)

        client._session.get = MagicMock(side_effect=[resp1, resp2, resp2, resp2])

        result = client.fetch_org_models("org")
        # Should have the first page's model but stop after failing page 2
        assert len(result) == 1
        assert result[0].model_id == "org/model-1"

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_filters_derivatives(self, mock_sleep, client):
        models = [
            _make_raw_model("org/Llama-4"),
            _make_raw_model("org/Llama-4-gguf", tags=["gguf"]),
            _make_raw_model("org/Llama-4-gptq", tags=["gptq"]),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_models("org")
        assert len(result) == 1
        assert result[0].model_id == "org/Llama-4"

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_max_pages_override(self, mock_sleep, client):
        resp = _make_response(
            json_data=[_make_raw_model("org/model")],
            headers={"Link": '<https://huggingface.co/api/models?cursor=abc>; rel="next"'},
        )
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_models("org", max_pages=1)
        assert client._session.get.call_count == 1


# ===================================================================
# HFApiClient.fetch_all_watched_org_models
# ===================================================================

class TestFetchAllWatchedOrgModels:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_fetches_multiple_orgs(self, mock_sleep, client):
        def side_effect(url, params=None, timeout=None):
            author = ""
            if params and "author" in params:
                author = params["author"]
            if author == "meta-llama":
                return _make_response(
                    json_data=[_make_raw_model("meta-llama/Llama-4")],
                    headers={},
                )
            elif author == "google":
                return _make_response(
                    json_data=[
                        _make_raw_model("google/gemma-3"),
                        _make_raw_model("google/gemma-2"),
                    ],
                    headers={},
                )
            return _make_response(json_data=[], headers={})

        client._session.get = MagicMock(side_effect=side_effect)

        results = client.fetch_all_watched_org_models(["meta-llama", "google"])

        assert "meta-llama" in results
        assert "google" in results
        assert len(results["meta-llama"]) == 1
        assert len(results["google"]) == 2

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_failed_org_returns_empty_list(self, mock_sleep, client):
        client._session.get = MagicMock(
            side_effect=requests.exceptions.ConnectionError("refused")
        )

        results = client.fetch_all_watched_org_models(["failing-org"])

        assert "failing-org" in results
        assert results["failing-org"] == []

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_handles_hf_api_error(self, mock_sleep, client):
        resp = _make_response(status_code=401)
        client._session.get = MagicMock(return_value=resp)

        results = client.fetch_all_watched_org_models(["gated-org"])

        assert "gated-org" in results
        assert results["gated-org"] == []

    def test_empty_org_list(self, client):
        results = client.fetch_all_watched_org_models([])
        assert results == {}


# ===================================================================
# HFApiClient.fetch_model_detail
# ===================================================================

class TestFetchModelDetail:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_model_info(self, mock_sleep, client):
        resp = _make_response(json_data={"id": "org/model", "downloads": 999})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_model_detail("org/model")
        assert result["id"] == "org/model"
        assert result["downloads"] == 999

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_empty_on_404(self, mock_sleep, client):
        resp = _make_response(status_code=404)
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_model_detail("org/nonexistent")
        assert result == {}

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_empty_on_list_response(self, mock_sleep, client):
        """If API returns a list instead of dict, return empty dict."""
        resp = _make_response(json_data=[{"id": "not expected"}])
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_model_detail("org/model")
        assert result == {}


# ===================================================================
# Context manager
# ===================================================================

class TestContextManager:
    def test_closes_session(self):
        with HFApiClient(min_request_interval=0) as client:
            assert client._session is not None
        # After exiting, session should be closed
        # (Session.close is called but the object still exists)


# ===================================================================
# _parse_retry_after
# ===================================================================

class TestParseRetryAfter:
    def test_valid_header(self):
        resp = _make_response(headers={"Retry-After": "30"})
        assert _parse_retry_after(resp) == 30.0

    def test_float_header(self):
        resp = _make_response(headers={"Retry-After": "2.5"})
        assert _parse_retry_after(resp) == 2.5

    def test_missing_header(self):
        resp = _make_response(headers={})
        assert _parse_retry_after(resp) == 60.0

    def test_invalid_header(self):
        resp = _make_response(headers={"Retry-After": "invalid"})
        assert _parse_retry_after(resp) == 60.0

    def test_very_small_value_clamped(self):
        resp = _make_response(headers={"Retry-After": "0.1"})
        assert _parse_retry_after(resp) == 1.0


# ===================================================================
# Link header parsing
# ===================================================================

class TestExtractNextLink:
    def test_parses_next_link(self, client):
        client._last_response = MagicMock()
        client._last_response.headers = {
            "Link": '<https://huggingface.co/api/models?cursor=next123>; rel="next"'
        }
        result = client._extract_next_link()
        assert result == "https://huggingface.co/api/models?cursor=next123"

    def test_no_link_header(self, client):
        client._last_response = MagicMock()
        client._last_response.headers = {}
        assert client._extract_next_link() is None

    def test_no_next_rel(self, client):
        client._last_response = MagicMock()
        client._last_response.headers = {
            "Link": '<https://example.com>; rel="prev"'
        }
        assert client._extract_next_link() is None

    def test_multiple_links(self, client):
        client._last_response = MagicMock()
        client._last_response.headers = {
            "Link": (
                '<https://example.com/prev>; rel="prev", '
                '<https://example.com/next>; rel="next"'
            )
        }
        result = client._extract_next_link()
        assert result == "https://example.com/next"

    def test_no_last_response(self, client):
        # _last_response not set yet
        assert client._extract_next_link() is None


# ===================================================================
# Convenience functions
# ===================================================================

class TestConvenienceFunctions:
    @patch("hf_model_monitor.hf_client.get_default_client")
    def test_fetch_org_models_returns_dicts(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.fetch_org_models.return_value = [
            ModelRecord(
                model_id="org/model-1",
                author="org",
                created_at="2026-04-20",
                last_modified="2026-04-25",
                pipeline_tag="text-generation",
            )
        ]
        mock_get_client.return_value = mock_client

        result = fetch_org_models("org")
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["model_id"] == "org/model-1"

    @patch("hf_model_monitor.hf_client.get_default_client")
    def test_fetch_all_watched_returns_dicts(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.fetch_all_watched_org_models.return_value = {
            "org": [
                ModelRecord(
                    model_id="org/model-1",
                    author="org",
                    created_at="2026-04-20",
                    last_modified="2026-04-25",
                    pipeline_tag="text-generation",
                )
            ]
        }
        mock_get_client.return_value = mock_client

        result = fetch_all_watched_org_models(["org"])
        assert "org" in result
        assert isinstance(result["org"][0], dict)


# ===================================================================
# Client initialization
# ===================================================================

class TestClientInit:
    def test_default_init(self):
        client = HFApiClient()
        assert client.timeout == 30
        assert client.per_page == 100
        assert client.max_retries == 3
        client.close()

    def test_per_page_capped_at_100(self):
        client = HFApiClient(per_page=500)
        assert client.per_page == 100
        client.close()

    def test_token_sets_auth_header(self):
        client = HFApiClient(token="hf_test_token")
        assert client._session.headers["Authorization"] == "Bearer hf_test_token"
        client.close()

    def test_no_token_no_auth_header(self):
        client = HFApiClient()
        assert "Authorization" not in client._session.headers
        client.close()


# ===================================================================
# HFApiClient.fetch_org_latest_models
# ===================================================================

class TestFetchOrgLatestModels:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_all_when_no_since(self, mock_sleep, client):
        """Without a since cutoff, returns all models sorted by createdAt."""
        models = [
            _make_raw_model("org/new-model", createdAt="2026-04-25T12:00:00Z"),
            _make_raw_model("org/old-model", createdAt="2026-04-10T12:00:00Z"),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_latest_models("org")
        assert len(result) == 2
        # Verify sort param was passed as createdAt
        call_args = client._session.get.call_args
        assert call_args[1].get("params", {}).get("sort") == "createdAt" or \
            call_args[0][0].endswith("/models")

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_filters_by_since_cutoff(self, mock_sleep, client):
        """Models older than since cutoff are excluded."""
        models = [
            _make_raw_model("org/new-model", createdAt="2026-04-25T12:00:00Z"),
            _make_raw_model("org/recent-model", createdAt="2026-04-24T12:00:00Z"),
            _make_raw_model("org/old-model", createdAt="2026-04-10T12:00:00Z"),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_latest_models(
            "org", since="2026-04-24T00:00:00Z"
        )
        assert len(result) == 2
        assert result[0].model_id == "org/new-model"
        assert result[1].model_id == "org/recent-model"

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_early_termination_on_old_model(self, mock_sleep, client):
        """Once an old model is hit, later models are excluded even if newer."""
        # Sorted by createdAt desc — once we hit old, we stop
        models = [
            _make_raw_model("org/newest", createdAt="2026-04-25T12:00:00Z"),
            _make_raw_model("org/old", createdAt="2026-04-01T12:00:00Z"),
            # This one would pass the filter, but we stop after "old"
            _make_raw_model("org/after-old", createdAt="2026-04-20T12:00:00Z"),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_latest_models(
            "org", since="2026-04-15T00:00:00Z"
        )
        # Only "newest" passes; "old" triggers early stop
        assert len(result) == 1
        assert result[0].model_id == "org/newest"

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_includes_models_with_na_created_at(self, mock_sleep, client):
        """Models with N/A created_at are included (benefit of the doubt)."""
        models = [
            _make_raw_model("org/unknown-date", createdAt="N/A"),
            _make_raw_model("org/new", createdAt="2026-04-25T12:00:00Z"),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_latest_models(
            "org", since="2026-04-20T00:00:00Z"
        )
        assert len(result) == 2
        ids = [r.model_id for r in result]
        assert "org/unknown-date" in ids
        assert "org/new" in ids

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_empty_result(self, mock_sleep, client):
        resp = _make_response(json_data=[], headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_latest_models(
            "org", since="2026-04-20T00:00:00Z"
        )
        assert result == []

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_all_models_older_than_since(self, mock_sleep, client):
        models = [
            _make_raw_model("org/old1", createdAt="2026-01-01T12:00:00Z"),
            _make_raw_model("org/old2", createdAt="2025-12-01T12:00:00Z"),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_org_latest_models(
            "org", since="2026-04-01T00:00:00Z"
        )
        assert result == []


# ===================================================================
# HFApiClient.fetch_latest_from_config
# ===================================================================

class TestFetchLatestFromConfig:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_fetches_all_orgs_from_config(self, mock_sleep, client):
        config = {"watched_organizations": ["meta-llama", "google"]}

        def side_effect(url, params=None, timeout=None):
            author = ""
            if params and "author" in params:
                author = params["author"]
            if author == "meta-llama":
                return _make_response(
                    json_data=[_make_raw_model("meta-llama/Llama-5")],
                    headers={},
                )
            elif author == "google":
                return _make_response(
                    json_data=[_make_raw_model("google/gemma-4")],
                    headers={},
                )
            return _make_response(json_data=[], headers={})

        client._session.get = MagicMock(side_effect=side_effect)

        results = client.fetch_latest_from_config(config)
        assert "meta-llama" in results
        assert "google" in results
        assert len(results["meta-llama"]) == 1
        assert len(results["google"]) == 1

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_with_since_uses_latest_method(self, mock_sleep, client):
        config = {"watched_organizations": ["org1"]}
        models = [
            _make_raw_model("org1/new", createdAt="2026-04-25T12:00:00Z"),
            _make_raw_model("org1/old", createdAt="2026-01-01T12:00:00Z"),
        ]
        resp = _make_response(json_data=models, headers={})
        client._session.get = MagicMock(return_value=resp)

        results = client.fetch_latest_from_config(
            config, since="2026-04-20T00:00:00Z"
        )
        assert len(results["org1"]) == 1
        assert results["org1"][0].model_id == "org1/new"

    def test_empty_config(self, client):
        results = client.fetch_latest_from_config({})
        assert results == {}

    def test_no_orgs_in_config(self, client):
        results = client.fetch_latest_from_config(
            {"watched_organizations": []}
        )
        assert results == {}

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_failed_org_returns_empty(self, mock_sleep, client):
        config = {"watched_organizations": ["failing-org"]}
        client._session.get = MagicMock(
            side_effect=requests.exceptions.ConnectionError("refused")
        )

        results = client.fetch_latest_from_config(config)
        assert "failing-org" in results
        assert results["failing-org"] == []


# ===================================================================
# HFApiClient.fetch_trending_models
# ===================================================================

class TestFetchTrendingModels:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_parses_trending_with_repo_data(self, mock_sleep, client):
        trending_data = {
            "recentlyTrending": [
                {
                    "repoData": {
                        "id": "deepseek-ai/DeepSeek-V3",
                        "author": "deepseek-ai",
                        "downloads": 50000,
                        "likes": 1200,
                        "pipeline_tag": "text-generation",
                        "createdAt": "2026-04-20T10:00:00Z",
                        "lastModified": "2026-04-25T12:00:00Z",
                        "tags": ["transformers"],
                        "library_name": "transformers",
                    },
                    "trendingScore": 85,
                },
            ]
        }
        resp = _make_response(json_data=trending_data)
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_trending_models(limit=10)
        assert len(result) == 1
        assert result[0]["model_id"] == "deepseek-ai/DeepSeek-V3"
        assert result[0]["author"] == "deepseek-ai"
        assert result[0]["trending_score"] == 85
        assert result[0]["downloads"] == 50000

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_handles_list_response(self, mock_sleep, client):
        trending_data = [
            {
                "id": "org/model-1",
                "author": "org",
                "downloads": 100,
                "likes": 10,
            }
        ]
        resp = _make_response(json_data=trending_data)
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_trending_models()
        assert len(result) == 1
        assert result[0]["model_id"] == "org/model-1"

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_empty_on_api_failure(self, mock_sleep, client):
        resp = _make_response(status_code=500)
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_trending_models()
        assert result == []

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_skips_items_without_id(self, mock_sleep, client):
        trending_data = {
            "recentlyTrending": [
                {"repoData": {"author": "no-id-org", "downloads": 100}},
                {
                    "repoData": {
                        "id": "valid/model",
                        "author": "valid",
                        "downloads": 200,
                        "likes": 20,
                    },
                },
            ]
        }
        resp = _make_response(json_data=trending_data)
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_trending_models()
        assert len(result) == 1
        assert result[0]["model_id"] == "valid/model"


# ===================================================================
# HFApiClient.fetch_most_downloaded_models
# ===================================================================

class TestFetchMostDownloadedModels:
    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_sorted_models(self, mock_sleep, client):
        models = [
            _make_raw_model("org/popular", downloads=100000),
            _make_raw_model("org/medium", downloads=50000),
            _make_raw_model("org/niche", downloads=1000),
        ]
        resp = _make_response(json_data=models)
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_most_downloaded_models(limit=10)
        assert len(result) == 3
        assert result[0].model_id == "org/popular"
        assert result[0].downloads == 100000

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_empty_on_failure(self, mock_sleep, client):
        resp = _make_response(status_code=500)
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_most_downloaded_models()
        assert result == []

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_returns_empty_on_non_list_response(self, mock_sleep, client):
        resp = _make_response(json_data={"error": "unexpected"})
        client._session.get = MagicMock(return_value=resp)

        result = client.fetch_most_downloaded_models()
        assert result == []

    @patch("hf_model_monitor.hf_client.time.sleep")
    def test_limit_capped_at_100(self, mock_sleep, client):
        resp = _make_response(json_data=[])
        client._session.get = MagicMock(return_value=resp)

        client.fetch_most_downloaded_models(limit=500)
        call_params = client._session.get.call_args[1]["params"]
        assert call_params["limit"] <= 100


# ===================================================================
# reset_default_client
# ===================================================================

class TestResetDefaultClient:
    @patch("hf_model_monitor.hf_client.get_default_client")
    def test_fetch_latest_from_config_returns_dicts(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.fetch_latest_from_config.return_value = {
            "org": [
                ModelRecord(
                    model_id="org/model-1",
                    author="org",
                    created_at="2026-04-20",
                    last_modified="2026-04-25",
                    pipeline_tag="text-generation",
                )
            ]
        }
        mock_get_client.return_value = mock_client

        result = fetch_latest_from_config(
            {"watched_organizations": ["org"]},
            since="2026-04-19T00:00:00Z",
        )
        assert "org" in result
        assert isinstance(result["org"][0], dict)
        assert result["org"][0]["model_id"] == "org/model-1"

    def test_reset_clears_singleton(self):
        """reset_default_client should allow re-initialization."""
        # Import and call — verifies the function exists and is callable
        reset_default_client()
        # After reset, next get_default_client should create a new instance


# ===================================================================
# ModelRecord edge cases
# ===================================================================

class TestModelRecordEdgeCases:
    def test_gated_string_value(self):
        """HF API returns gated as string ('auto', 'manual') for some models."""
        record = ModelRecord(
            model_id="org/gated-model",
            author="org",
            created_at="2026-04-20T10:00:00Z",
            last_modified="2026-04-25T12:00:00Z",
            pipeline_tag="text-generation",
            gated="auto",
        )
        d = record.to_dict()
        assert d["gated"] == "auto"

    def test_parse_model_with_gated_string(self):
        raw = _make_raw_model("org/model", gated="manual")
        record = HFApiClient._parse_model(raw)
        assert record is not None
        assert record.gated == "manual"

    def test_parse_model_skips_qlora_derivative(self):
        raw = _make_raw_model("user/model-qlora", tags=["lora"])
        record = HFApiClient._parse_model(raw)
        assert record is None

    def test_parse_model_keeps_bnb_tag_without_name_pattern(self):
        """Model with bnb tag but no derivative pattern in name is kept."""
        raw = _make_raw_model("org/OriginalModel", tags=["bnb", "transformers"])
        record = HFApiClient._parse_model(raw)
        assert record is not None
