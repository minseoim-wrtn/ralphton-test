from unittest.mock import patch, MagicMock

from hf_model_monitor.hf_fetcher import fetch_trending_models, fetch_model_info


class TestFetchTrendingModels:
    @patch("hf_model_monitor.hf_fetcher.requests.get")
    def test_returns_parsed_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "recentlyTrending": [
                {
                    "repoData": {
                        "id": "deepseek-ai/DeepSeek-V3",
                        "author": "deepseek-ai",
                        "downloads": 50000,
                        "likes": 1200,
                    }
                },
                {
                    "repoData": {
                        "id": "meta-llama/Llama-4",
                        "author": "meta-llama",
                        "downloads": 80000,
                        "likes": 2000,
                    }
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_trending_models(limit=10)

        assert len(result) == 2
        assert result[0]["model_id"] == "deepseek-ai/DeepSeek-V3"
        assert result[0]["author"] == "deepseek-ai"
        assert result[0]["downloads"] == 50000
        assert result[0]["likes"] == 1200
        assert result[1]["model_id"] == "meta-llama/Llama-4"

    @patch("hf_model_monitor.hf_fetcher.requests.get")
    def test_handles_list_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": "org/model-1", "author": "org", "downloads": 100, "likes": 10}
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_trending_models()
        assert len(result) == 1
        assert result[0]["model_id"] == "org/model-1"

    @patch("hf_model_monitor.hf_fetcher.requests.get")
    def test_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = fetch_trending_models()
        assert result == []

    @patch("hf_model_monitor.hf_fetcher.requests.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        mock_get.return_value = mock_resp

        result = fetch_trending_models()
        assert result == []


class TestFetchModelInfo:
    @patch("hf_model_monitor.hf_fetcher.requests.get")
    def test_returns_model_info(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "org/model", "downloads": 999}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_model_info("org/model")
        assert result["id"] == "org/model"

    @patch("hf_model_monitor.hf_fetcher.requests.get")
    def test_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("fail")
        result = fetch_model_info("org/model")
        assert result == {}
