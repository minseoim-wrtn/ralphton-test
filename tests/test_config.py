import os
import tempfile

import yaml

from hf_model_monitor.config import (
    DEFAULT_WATCHED_ORGS,
    add_organization,
    get_watched_organizations,
    list_organizations,
    load_config,
    remove_organization,
    save_config,
    validate_org_name,
    validate_watched_orgs,
)


# ---------------------------------------------------------------------------
# validate_org_name
# ---------------------------------------------------------------------------
class TestValidateOrgName:
    def test_valid_simple_names(self):
        assert validate_org_name("meta-llama") is True
        assert validate_org_name("google") is True
        assert validate_org_name("deepseek-ai") is True

    def test_valid_names_with_dots_and_underscores(self):
        assert validate_org_name("org.name") is True
        assert validate_org_name("org_name") is True
        assert validate_org_name("Org.Name_123") is True

    def test_valid_single_char(self):
        assert validate_org_name("a") is True
        assert validate_org_name("Z") is True
        assert validate_org_name("0") is True

    def test_valid_two_char(self):
        assert validate_org_name("ab") is True
        assert validate_org_name("A1") is True

    def test_valid_mixed_case(self):
        assert validate_org_name("CohereForAI") is True
        assert validate_org_name("EleutherAI") is True
        assert validate_org_name("Qwen") is True

    def test_invalid_empty_string(self):
        assert validate_org_name("") is False

    def test_invalid_none(self):
        assert validate_org_name(None) is False

    def test_invalid_non_string(self):
        assert validate_org_name(123) is False
        assert validate_org_name(["meta-llama"]) is False

    def test_invalid_leading_hyphen(self):
        assert validate_org_name("-meta") is False

    def test_invalid_trailing_hyphen(self):
        assert validate_org_name("meta-") is False

    def test_invalid_special_chars(self):
        assert validate_org_name("org/name") is False
        assert validate_org_name("org name") is False
        assert validate_org_name("org@name") is False
        assert validate_org_name("org!") is False


# ---------------------------------------------------------------------------
# validate_watched_orgs
# ---------------------------------------------------------------------------
class TestValidateWatchedOrgs:
    def test_all_valid(self):
        orgs = ["meta-llama", "google", "Qwen"]
        valid, invalid = validate_watched_orgs(orgs)
        assert valid == ["meta-llama", "google", "Qwen"]
        assert invalid == []

    def test_filters_invalid(self):
        orgs = ["meta-llama", "-bad-name", "google"]
        valid, invalid = validate_watched_orgs(orgs)
        assert valid == ["meta-llama", "google"]
        assert invalid == ["-bad-name"]

    def test_deduplicates(self):
        orgs = ["meta-llama", "google", "meta-llama", "google"]
        valid, invalid = validate_watched_orgs(orgs)
        assert valid == ["meta-llama", "google"]
        assert invalid == []

    def test_strips_whitespace(self):
        orgs = ["  meta-llama  ", "google"]
        valid, invalid = validate_watched_orgs(orgs)
        assert valid == ["meta-llama", "google"]

    def test_skips_empty_strings(self):
        orgs = ["meta-llama", "", "  ", "google"]
        valid, invalid = validate_watched_orgs(orgs)
        assert valid == ["meta-llama", "google"]

    def test_handles_non_string_items(self):
        orgs = ["meta-llama", 123, None, "google"]
        valid, invalid = validate_watched_orgs(orgs)
        assert valid == ["meta-llama", "google"]
        assert "123" in invalid
        assert "None" in invalid

    def test_empty_list(self):
        valid, invalid = validate_watched_orgs([])
        assert valid == []
        assert invalid == []

    def test_preserves_order(self):
        orgs = ["Qwen", "meta-llama", "google", "deepseek-ai"]
        valid, _ = validate_watched_orgs(orgs)
        assert valid == ["Qwen", "meta-llama", "google", "deepseek-ai"]


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_returns_defaults_when_file_missing(self):
        cfg = load_config("/nonexistent/settings.yaml")
        assert cfg["watched_organizations"] == DEFAULT_WATCHED_ORGS
        assert cfg["polling_interval_hours"] == 12
        assert cfg["slack_webhook_url"] == ""

    def test_loads_valid_yaml(self):
        data = {
            "watched_organizations": ["meta-llama", "google"],
            "polling_interval_hours": 12,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["watched_organizations"] == ["meta-llama", "google"]
            assert cfg["polling_interval_hours"] == 12
        finally:
            os.unlink(path)

    def test_falls_back_on_corrupt_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(": : : invalid yaml {{{")
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["watched_organizations"] == DEFAULT_WATCHED_ORGS
        finally:
            os.unlink(path)

    def test_falls_back_on_non_dict_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(["just", "a", "list"], f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["watched_organizations"] == DEFAULT_WATCHED_ORGS
        finally:
            os.unlink(path)

    def test_skips_invalid_orgs_in_yaml(self):
        data = {"watched_organizations": ["meta-llama", "-invalid-", "google"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["watched_organizations"] == ["meta-llama", "google"]
        finally:
            os.unlink(path)

    def test_falls_back_when_all_orgs_invalid(self):
        data = {"watched_organizations": ["-bad-", "also-bad-"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["watched_organizations"] == DEFAULT_WATCHED_ORGS
        finally:
            os.unlink(path)

    def test_falls_back_when_orgs_not_a_list(self):
        data = {"watched_organizations": "meta-llama"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["watched_organizations"] == DEFAULT_WATCHED_ORGS
        finally:
            os.unlink(path)

    def test_falls_back_when_orgs_empty_list(self):
        data = {"watched_organizations": []}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["watched_organizations"] == DEFAULT_WATCHED_ORGS
        finally:
            os.unlink(path)

    def test_ignores_invalid_polling_interval(self):
        data = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": -5,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["polling_interval_hours"] == 12  # default
        finally:
            os.unlink(path)

    def test_ignores_non_numeric_polling_interval(self):
        data = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": "fast",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["polling_interval_hours"] == 12
        finally:
            os.unlink(path)

    def test_env_var_overrides_yaml_slack_url(self):
        data = {"slack_webhook_url": "https://yaml-url.example.com"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            os.environ["SLACK_WEBHOOK_URL"] = "https://env-url.example.com"
            cfg = load_config(path)
            assert cfg["slack_webhook_url"] == "https://env-url.example.com"
        finally:
            os.environ.pop("SLACK_WEBHOOK_URL", None)
            os.unlink(path)

    def test_yaml_slack_url_used_when_no_env(self):
        data = {"slack_webhook_url": "https://yaml-url.example.com"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            os.environ.pop("SLACK_WEBHOOK_URL", None)
            cfg = load_config(path)
            assert cfg["slack_webhook_url"] == "https://yaml-url.example.com"
        finally:
            os.unlink(path)

    def test_missing_keys_get_defaults(self):
        data = {"watched_organizations": ["meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            os.environ.pop("SLACK_WEBHOOK_URL", None)
            cfg = load_config(path)
            assert cfg["polling_interval_hours"] == 12
            assert cfg["slack_webhook_url"] == ""
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_watched_organizations
# ---------------------------------------------------------------------------
class TestGetWatchedOrganizations:
    def test_returns_list_from_config(self):
        data = {"watched_organizations": ["meta-llama", "google"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            orgs = get_watched_organizations(path)
            assert orgs == ["meta-llama", "google"]
        finally:
            os.unlink(path)

    def test_returns_defaults_when_no_file(self):
        orgs = get_watched_organizations("/nonexistent/settings.yaml")
        assert orgs == DEFAULT_WATCHED_ORGS

    def test_default_orgs_are_all_valid(self):
        """Ensure every built-in default org passes validation."""
        for org in DEFAULT_WATCHED_ORGS:
            assert validate_org_name(org), f"Default org '{org}' is invalid"

    def test_default_orgs_have_no_duplicates(self):
        assert len(DEFAULT_WATCHED_ORGS) == len(set(DEFAULT_WATCHED_ORGS))


# ---------------------------------------------------------------------------
# trending_thresholds
# ---------------------------------------------------------------------------
class TestTrendingThresholds:
    """Tests for the trending_thresholds config section."""

    # -- defaults --

    def test_defaults_present_when_file_missing(self):
        cfg = load_config("/nonexistent/settings.yaml")
        th = cfg["trending_thresholds"]
        assert th["enabled"] is False
        assert th["download_surge_count"] == 10000
        assert th["trending_score"] == 50
        assert th["time_window_hours"] == 24

    def test_defaults_present_when_section_omitted(self):
        data = {"watched_organizations": ["meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            th = cfg["trending_thresholds"]
            assert th["enabled"] is False
            assert th["download_surge_count"] == 10000
            assert th["trending_score"] == 50
            assert th["time_window_hours"] == 24
        finally:
            os.unlink(path)

    # -- full override --

    def test_full_override(self):
        data = {
            "trending_thresholds": {
                "enabled": True,
                "download_surge_count": 5000,
                "trending_score": 75,
                "time_window_hours": 48,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            th = cfg["trending_thresholds"]
            assert th["enabled"] is True
            assert th["download_surge_count"] == 5000
            assert th["trending_score"] == 75
            assert th["time_window_hours"] == 48
        finally:
            os.unlink(path)

    # -- partial override --

    def test_partial_override_keeps_remaining_defaults(self):
        data = {
            "trending_thresholds": {
                "enabled": True,
                "download_surge_count": 20000,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            th = cfg["trending_thresholds"]
            assert th["enabled"] is True
            assert th["download_surge_count"] == 20000
            # unchanged defaults
            assert th["trending_score"] == 50
            assert th["time_window_hours"] == 24
        finally:
            os.unlink(path)

    # -- enabled validation --

    def test_enabled_rejects_non_bool(self):
        data = {"trending_thresholds": {"enabled": "yes"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["enabled"] is False  # default
        finally:
            os.unlink(path)

    # -- download_surge_count validation --

    def test_download_surge_count_rejects_zero(self):
        data = {"trending_thresholds": {"download_surge_count": 0}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["download_surge_count"] == 10000
        finally:
            os.unlink(path)

    def test_download_surge_count_rejects_negative(self):
        data = {"trending_thresholds": {"download_surge_count": -500}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["download_surge_count"] == 10000
        finally:
            os.unlink(path)

    def test_download_surge_count_rejects_string(self):
        data = {"trending_thresholds": {"download_surge_count": "many"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["download_surge_count"] == 10000
        finally:
            os.unlink(path)

    def test_download_surge_count_accepts_float(self):
        data = {"trending_thresholds": {"download_surge_count": 7500.5}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["download_surge_count"] == 7500
        finally:
            os.unlink(path)

    # -- trending_score validation --

    def test_trending_score_accepts_zero(self):
        data = {"trending_thresholds": {"trending_score": 0}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["trending_score"] == 0
        finally:
            os.unlink(path)

    def test_trending_score_rejects_negative(self):
        data = {"trending_thresholds": {"trending_score": -10}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["trending_score"] == 50
        finally:
            os.unlink(path)

    def test_trending_score_rejects_bool(self):
        """Booleans are a subclass of int in Python; they should be rejected."""
        data = {"trending_thresholds": {"trending_score": True}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["trending_score"] == 50
        finally:
            os.unlink(path)

    # -- time_window_hours validation --

    def test_time_window_hours_accepts_boundary_min(self):
        data = {"trending_thresholds": {"time_window_hours": 1}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["time_window_hours"] == 1
        finally:
            os.unlink(path)

    def test_time_window_hours_accepts_boundary_max(self):
        data = {"trending_thresholds": {"time_window_hours": 168}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["time_window_hours"] == 168
        finally:
            os.unlink(path)

    def test_time_window_hours_rejects_zero(self):
        data = {"trending_thresholds": {"time_window_hours": 0}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["time_window_hours"] == 24
        finally:
            os.unlink(path)

    def test_time_window_hours_rejects_over_168(self):
        data = {"trending_thresholds": {"time_window_hours": 200}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["time_window_hours"] == 24
        finally:
            os.unlink(path)

    # -- section-level validation --

    def test_non_dict_section_uses_defaults(self):
        data = {"trending_thresholds": "invalid"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            th = cfg["trending_thresholds"]
            assert th["enabled"] is False
            assert th["download_surge_count"] == 10000
            assert th["trending_score"] == 50
            assert th["time_window_hours"] == 24
        finally:
            os.unlink(path)

    def test_list_section_uses_defaults(self):
        data = {"trending_thresholds": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg["trending_thresholds"]["enabled"] is False
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------
class TestSaveConfig:
    def test_saves_and_roundtrips(self):
        cfg = {
            "watched_organizations": ["meta-llama", "google"],
            "polling_interval_hours": 6,
            "run_on_startup": False,
            "slack_webhook_url": "https://hooks.slack.com/test",
            "trending_thresholds": {
                "enabled": True,
                "download_surge_count": 5000,
                "trending_score": 75,
                "time_window_hours": 48,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded["watched_organizations"] == ["meta-llama", "google"]
            assert loaded["polling_interval_hours"] == 6
            assert loaded["run_on_startup"] is False
            assert loaded["trending_thresholds"]["enabled"] is True
            assert loaded["trending_thresholds"]["download_surge_count"] == 5000
        finally:
            os.unlink(path)

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir", "settings.yaml")
            cfg = {
                "watched_organizations": ["meta-llama"],
                "polling_interval_hours": 12,
                "run_on_startup": True,
                "slack_webhook_url": "",
                "trending_thresholds": {
                    "enabled": False,
                    "download_surge_count": 10000,
                    "trending_score": 50,
                    "time_window_hours": 24,
                },
            }
            save_config(cfg, nested)
            assert os.path.exists(nested)
            loaded = load_config(nested)
            assert loaded["watched_organizations"] == ["meta-llama"]

    def test_writes_header_comment(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            cfg = {
                "watched_organizations": ["meta-llama"],
                "polling_interval_hours": 12,
                "run_on_startup": True,
                "slack_webhook_url": "",
                "trending_thresholds": {
                    "enabled": False,
                    "download_surge_count": 10000,
                    "trending_score": 50,
                    "time_window_hours": 24,
                },
            }
            save_config(cfg, path)
            with open(path) as f:
                content = f.read()
            assert "HuggingFace Model Monitor" in content
            assert content.startswith("#")
        finally:
            os.unlink(path)

    def test_preserves_defaults_for_missing_keys(self):
        """save_config fills in defaults for any keys missing from the dict."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            save_config({}, path)
            loaded = load_config(path)
            assert loaded["watched_organizations"] == list(DEFAULT_WATCHED_ORGS)
            assert loaded["polling_interval_hours"] == 12
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# add_organization
# ---------------------------------------------------------------------------
class TestAddOrganization:
    def test_adds_new_org(self):
        data = {"watched_organizations": ["meta-llama", "google"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = add_organization("mistralai", path)
            assert ok is True
            assert "added" in msg.lower()
            orgs = get_watched_organizations(path)
            assert "mistralai" in orgs
            assert orgs == ["meta-llama", "google", "mistralai"]
        finally:
            os.unlink(path)

    def test_duplicate_org_is_idempotent(self):
        data = {"watched_organizations": ["meta-llama", "google"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = add_organization("google", path)
            assert ok is True
            assert "already" in msg.lower()
            orgs = get_watched_organizations(path)
            assert orgs.count("google") == 1
        finally:
            os.unlink(path)

    def test_rejects_invalid_org_name(self):
        data = {"watched_organizations": ["meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = add_organization("-invalid-", path)
            assert ok is False
            assert "invalid" in msg.lower()
            orgs = get_watched_organizations(path)
            assert "-invalid-" not in orgs
        finally:
            os.unlink(path)

    def test_rejects_empty_string(self):
        ok, msg = add_organization("", "/tmp/dummy.yaml")
        assert ok is False
        assert "non-empty" in msg.lower()

    def test_rejects_none(self):
        ok, msg = add_organization(None, "/tmp/dummy.yaml")
        assert ok is False

    def test_rejects_whitespace_only(self):
        ok, msg = add_organization("   ", "/tmp/dummy.yaml")
        assert ok is False

    def test_strips_whitespace_before_adding(self):
        data = {"watched_organizations": ["meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = add_organization("  google  ", path)
            assert ok is True
            orgs = get_watched_organizations(path)
            assert "google" in orgs
        finally:
            os.unlink(path)

    def test_persists_across_reload(self):
        """Verify the add is durable — survives a fresh load_config call."""
        data = {"watched_organizations": ["meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            add_organization("deepseek-ai", path)
            # Fresh load from disk
            cfg = load_config(path)
            assert "deepseek-ai" in cfg["watched_organizations"]
            assert "meta-llama" in cfg["watched_organizations"]
        finally:
            os.unlink(path)

    def test_preserves_other_config_keys(self):
        data = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": 6,
            "slack_webhook_url": "https://hooks.slack.com/test",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            os.environ.pop("SLACK_WEBHOOK_URL", None)
            add_organization("google", path)
            cfg = load_config(path)
            assert cfg["polling_interval_hours"] == 6
            assert cfg["slack_webhook_url"] == "https://hooks.slack.com/test"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# remove_organization
# ---------------------------------------------------------------------------
class TestRemoveOrganization:
    def test_removes_existing_org(self):
        data = {"watched_organizations": ["meta-llama", "google", "mistralai"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = remove_organization("google", path)
            assert ok is True
            assert "removed" in msg.lower()
            orgs = get_watched_organizations(path)
            assert "google" not in orgs
            assert orgs == ["meta-llama", "mistralai"]
        finally:
            os.unlink(path)

    def test_rejects_nonexistent_org(self):
        data = {"watched_organizations": ["meta-llama", "google"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = remove_organization("nonexistent-org", path)
            assert ok is False
            assert "not in" in msg.lower()
        finally:
            os.unlink(path)

    def test_prevents_removing_last_org(self):
        data = {"watched_organizations": ["meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = remove_organization("meta-llama", path)
            assert ok is False
            assert "last" in msg.lower()
            # Org should still be there
            orgs = get_watched_organizations(path)
            assert orgs == ["meta-llama"]
        finally:
            os.unlink(path)

    def test_rejects_empty_string(self):
        ok, msg = remove_organization("", "/tmp/dummy.yaml")
        assert ok is False
        assert "non-empty" in msg.lower()

    def test_rejects_none(self):
        ok, msg = remove_organization(None, "/tmp/dummy.yaml")
        assert ok is False

    def test_case_sensitive_removal(self):
        """HuggingFace IDs are case-sensitive: 'Qwen' != 'qwen'."""
        data = {"watched_organizations": ["Qwen", "meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            ok, msg = remove_organization("qwen", path)
            assert ok is False  # wrong case
            orgs = get_watched_organizations(path)
            assert "Qwen" in orgs
        finally:
            os.unlink(path)

    def test_persists_removal_across_reload(self):
        data = {"watched_organizations": ["meta-llama", "google", "Qwen"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            remove_organization("google", path)
            cfg = load_config(path)
            assert "google" not in cfg["watched_organizations"]
            assert "meta-llama" in cfg["watched_organizations"]
            assert "Qwen" in cfg["watched_organizations"]
        finally:
            os.unlink(path)

    def test_preserves_other_config_keys(self):
        data = {
            "watched_organizations": ["meta-llama", "google"],
            "polling_interval_hours": 8,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            remove_organization("google", path)
            cfg = load_config(path)
            assert cfg["polling_interval_hours"] == 8
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# list_organizations
# ---------------------------------------------------------------------------
class TestListOrganizations:
    def test_returns_same_as_get_watched(self):
        data = {"watched_organizations": ["meta-llama", "google"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            assert list_organizations(path) == get_watched_organizations(path)
        finally:
            os.unlink(path)

    def test_returns_defaults_when_no_file(self):
        orgs = list_organizations("/nonexistent/settings.yaml")
        assert orgs == DEFAULT_WATCHED_ORGS

    def test_reflects_add(self):
        data = {"watched_organizations": ["meta-llama"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            add_organization("google", path)
            orgs = list_organizations(path)
            assert orgs == ["meta-llama", "google"]
        finally:
            os.unlink(path)

    def test_reflects_remove(self):
        data = {"watched_organizations": ["meta-llama", "google", "Qwen"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            remove_organization("google", path)
            orgs = list_organizations(path)
            assert orgs == ["meta-llama", "Qwen"]
        finally:
            os.unlink(path)
