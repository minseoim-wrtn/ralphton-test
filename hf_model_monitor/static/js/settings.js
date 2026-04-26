/**
 * HF Model Monitor — Settings Page Client
 *
 * Shared API hooks (useSettings pattern) for loading and persisting
 * settings state from the backend. No external dependencies — plain
 * vanilla JS for PM maintainability.
 *
 * Architecture:
 *   useSettings()  — singleton hook: load, get, set, save, discard
 *   Sections:      — each section binds its DOM controls to the hook
 *   Save bar:      — tracks dirty state; save/discard buttons
 *
 * Data flow:
 *   1. Page load  → useSettings.load() → GET /api/settings
 *   2. User edits → useSettings.set(key, value) → dirty flag
 *   3. Save       → useSettings.save() → PUT /api/settings
 *   4. Discard    → useSettings.discard() → revert to last-saved state
 */

(function () {
    "use strict";

    // -----------------------------------------------------------------------
    // useSettings — shared settings state hook
    //
    // Provides a centralized way to load, read, mutate, and persist
    // settings. All section renderers read from and write to this hook.
    // -----------------------------------------------------------------------
    var useSettings = (function () {
        var _savedState = null;   // Last state from server (deep copy)
        var _currentState = null; // Working copy with user edits
        var _dirty = false;       // Whether current differs from saved
        var _loading = false;
        var _listeners = [];      // onChange callbacks

        /**
         * Register a callback invoked whenever state changes.
         * Callback receives (currentState, isDirty).
         */
        function onChange(fn) {
            _listeners.push(fn);
        }

        function _notify() {
            for (var i = 0; i < _listeners.length; i++) {
                _listeners[i](_currentState, _dirty);
            }
        }

        /**
         * Compute dirty flag by comparing current state to saved state.
         */
        function _computeDirty() {
            if (!_savedState || !_currentState) {
                _dirty = false;
                return;
            }
            _dirty = JSON.stringify(_currentState) !== JSON.stringify(_savedState);
        }

        /**
         * Load settings from the backend.
         * Returns a Promise that resolves with the settings object.
         */
        function load() {
            _loading = true;
            return fetch("/api/settings")
                .then(function (res) {
                    if (!res.ok) throw new Error("Failed to load settings: " + res.status);
                    return res.json();
                })
                .then(function (data) {
                    _savedState = _deepCopy(data);
                    _currentState = _deepCopy(data);
                    _dirty = false;
                    _loading = false;
                    _notify();
                    return _currentState;
                })
                .catch(function (err) {
                    _loading = false;
                    throw err;
                });
        }

        /**
         * Get the current (possibly edited) value of a settings key.
         * Supports dot notation: get("trending_thresholds.enabled")
         */
        function get(key) {
            if (!_currentState) return undefined;
            return _deepGet(_currentState, key);
        }

        /**
         * Get the entire current settings state (shallow ref).
         */
        function getAll() {
            return _currentState;
        }

        /**
         * Set a value in the working state.
         * Supports dot notation: set("trending_thresholds.enabled", true)
         * Triggers dirty check and notifies listeners.
         */
        function set(key, value) {
            if (!_currentState) return;
            _deepSet(_currentState, key, value);
            _computeDirty();
            _notify();
        }

        /**
         * Save all current settings to the backend via PUT /api/settings.
         * Returns a Promise that resolves with the server response.
         */
        function save() {
            if (!_currentState) return Promise.reject(new Error("No state to save"));

            return fetch("/api/settings", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(_currentState),
            })
                .then(function (res) {
                    if (!res.ok) {
                        return res.json().then(function (data) {
                            throw new Error(data.error || "Save failed: " + res.status);
                        });
                    }
                    return res.json();
                })
                .then(function (data) {
                    // Update saved state from server response (canonical)
                    var serverConfig = data.config || _currentState;
                    _savedState = _deepCopy(serverConfig);
                    _currentState = _deepCopy(serverConfig);
                    _dirty = false;
                    _notify();
                    return data;
                });
        }

        /**
         * Discard unsaved changes — revert to last-saved state.
         */
        function discard() {
            if (!_savedState) return;
            _currentState = _deepCopy(_savedState);
            _dirty = false;
            _notify();
        }

        /**
         * Check if settings have unsaved changes.
         */
        function isDirty() {
            return _dirty;
        }

        /**
         * Check if settings are currently loading.
         */
        function isLoading() {
            return _loading;
        }

        // --- Deep-access helpers ---

        function _deepCopy(obj) {
            return JSON.parse(JSON.stringify(obj));
        }

        function _deepGet(obj, path) {
            var keys = path.split(".");
            var cur = obj;
            for (var i = 0; i < keys.length; i++) {
                if (cur === undefined || cur === null) return undefined;
                cur = cur[keys[i]];
            }
            return cur;
        }

        function _deepSet(obj, path, value) {
            var keys = path.split(".");
            var cur = obj;
            for (var i = 0; i < keys.length - 1; i++) {
                if (cur[keys[i]] === undefined || cur[keys[i]] === null) {
                    cur[keys[i]] = {};
                }
                cur = cur[keys[i]];
            }
            cur[keys[keys.length - 1]] = value;
        }

        return {
            load: load,
            get: get,
            getAll: getAll,
            set: set,
            save: save,
            discard: discard,
            isDirty: isDirty,
            isLoading: isLoading,
            onChange: onChange,
        };
    })();

    // Expose useSettings globally for testability
    window.useSettings = useSettings;

    // -----------------------------------------------------------------------
    // DOM refs
    // -----------------------------------------------------------------------
    var settingsLoading = document.getElementById("settings-loading");
    var settingsContainer = document.getElementById("settings-container");
    var toast = document.getElementById("toast");
    var toastMessage = document.getElementById("toast-message");

    // General
    var pollingInterval = document.getElementById("polling-interval");
    var runOnStartup = document.getElementById("run-on-startup");

    // Slack
    var slackWebhook = document.getElementById("slack-webhook");
    var btnTestSlack = document.getElementById("btn-test-slack");
    var slackTestStatus = document.getElementById("slack-test-status");

    // Organizations
    var orgAddInput = document.getElementById("org-add-input");
    var btnAddOrg = document.getElementById("btn-add-org");
    var orgList = document.getElementById("org-list");
    var orgCount = document.getElementById("org-count");

    // Thresholds
    var trendingEnabled = document.getElementById("trending-enabled");
    var downloadSurge = document.getElementById("download-surge");
    var trendingScore = document.getElementById("trending-score");
    var timeWindow = document.getElementById("time-window");

    // Schema
    var schemaFields = document.getElementById("schema-fields");

    // Save bar
    var saveBar = document.getElementById("save-bar");
    var saveStatus = document.getElementById("save-status");
    var btnDiscard = document.getElementById("btn-discard");
    var btnSave = document.getElementById("btn-save");

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------
    document.addEventListener("DOMContentLoaded", function () {
        bindSaveBar();
        bindGeneralEvents();
        bindSlackEvents();
        bindOrgEvents();
        bindThresholdEvents();

        // Listen for state changes to update save bar
        useSettings.onChange(function (_state, dirty) {
            updateSaveBar(dirty);
        });

        // Load settings and populate UI
        useSettings.load()
            .then(function () {
                populateAll();
                settingsLoading.style.display = "none";
                settingsContainer.style.display = "";
            })
            .catch(function (err) {
                console.error("Failed to load settings:", err);
                settingsLoading.innerHTML =
                    '<p style="color: var(--color-danger);">Failed to load settings. Is the server running?</p>';
            });
    });

    // -----------------------------------------------------------------------
    // Populate all sections from current state
    // -----------------------------------------------------------------------
    function populateAll() {
        populateGeneral();
        populateSlack();
        populateOrgs();
        populateThresholds();
        populateSchema();
    }

    // -----------------------------------------------------------------------
    // Section: General
    // -----------------------------------------------------------------------
    function populateGeneral() {
        pollingInterval.value = useSettings.get("polling_interval_hours") || 12;
        runOnStartup.checked = useSettings.get("run_on_startup") !== false;
    }

    function bindGeneralEvents() {
        pollingInterval.addEventListener("input", function () {
            var val = parseInt(pollingInterval.value, 10);
            if (!isNaN(val) && val > 0) {
                useSettings.set("polling_interval_hours", val);
            }
        });

        runOnStartup.addEventListener("change", function () {
            useSettings.set("run_on_startup", runOnStartup.checked);
        });
    }

    // -----------------------------------------------------------------------
    // Section: Slack
    // -----------------------------------------------------------------------
    function populateSlack() {
        slackWebhook.value = useSettings.get("slack_webhook_url") || "";
        slackTestStatus.textContent = "";
    }

    function bindSlackEvents() {
        slackWebhook.addEventListener("input", function () {
            useSettings.set("slack_webhook_url", slackWebhook.value.trim());
        });

        btnTestSlack.addEventListener("click", function () {
            var url = slackWebhook.value.trim();
            if (!url) {
                slackTestStatus.textContent = "Enter a webhook URL first";
                slackTestStatus.className = "test-status test-error";
                return;
            }
            if (!url.startsWith("https://")) {
                slackTestStatus.textContent = "URL must start with https://";
                slackTestStatus.className = "test-status test-error";
                return;
            }
            slackTestStatus.textContent = "Testing...";
            slackTestStatus.className = "test-status test-pending";

            // Attempt a lightweight POST to verify the webhook responds
            fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: "HF Model Monitor — webhook test successful!",
                }),
                mode: "no-cors",  // Slack webhooks don't support CORS
            })
                .then(function () {
                    // no-cors always returns opaque response, so we can't check status
                    slackTestStatus.textContent = "Request sent (check Slack channel)";
                    slackTestStatus.className = "test-status test-success";
                })
                .catch(function (err) {
                    slackTestStatus.textContent = "Failed: " + err.message;
                    slackTestStatus.className = "test-status test-error";
                });
        });
    }

    // -----------------------------------------------------------------------
    // Section: Organizations
    // -----------------------------------------------------------------------
    function populateOrgs() {
        var orgs = useSettings.get("watched_organizations") || [];
        renderOrgList(orgs);
    }

    function renderOrgList(orgs) {
        var html = "";
        for (var i = 0; i < orgs.length; i++) {
            html +=
                '<div class="org-item" data-org="' + escapeHtml(orgs[i]) + '">' +
                '<span class="org-name">' + escapeHtml(orgs[i]) + "</span>" +
                '<a class="org-hf-link" href="https://huggingface.co/' +
                    escapeHtml(orgs[i]) +
                    '" target="_blank" rel="noopener" title="View on HuggingFace">HF</a>' +
                '<button class="org-remove" type="button" title="Remove organization" ' +
                    'data-org="' + escapeHtml(orgs[i]) + '">&times;</button>' +
                "</div>";
        }
        orgList.innerHTML = html;
        orgCount.textContent = orgs.length + " organization" + (orgs.length !== 1 ? "s" : "");

        // Bind remove buttons
        var removeBtns = orgList.querySelectorAll(".org-remove");
        for (var j = 0; j < removeBtns.length; j++) {
            removeBtns[j].addEventListener("click", handleRemoveOrg);
        }
    }

    function handleRemoveOrg(e) {
        var orgName = e.currentTarget.getAttribute("data-org");
        var orgs = useSettings.get("watched_organizations") || [];

        if (orgs.length <= 1) {
            showToast("Cannot remove the last organization", "error");
            return;
        }

        var filtered = orgs.filter(function (o) { return o !== orgName; });
        useSettings.set("watched_organizations", filtered);
        renderOrgList(filtered);
    }

    function bindOrgEvents() {
        btnAddOrg.addEventListener("click", addOrg);
        orgAddInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") {
                e.preventDefault();
                addOrg();
            }
        });
    }

    function addOrg() {
        var name = orgAddInput.value.trim();
        if (!name) return;

        // Basic client-side validation (same regex as backend)
        if (!/^[a-zA-Z0-9]([a-zA-Z0-9._-]{0,94}[a-zA-Z0-9])?$/.test(name)) {
            showToast("Invalid organization name: " + name, "error");
            return;
        }

        var orgs = useSettings.get("watched_organizations") || [];

        // Duplicate check
        if (orgs.indexOf(name) !== -1) {
            showToast("'" + name + "' is already in the list", "warning");
            orgAddInput.value = "";
            return;
        }

        orgs.push(name);
        useSettings.set("watched_organizations", orgs.slice());
        renderOrgList(orgs);
        orgAddInput.value = "";
        orgAddInput.focus();
    }

    // -----------------------------------------------------------------------
    // Section: Thresholds
    // -----------------------------------------------------------------------
    function populateThresholds() {
        var th = useSettings.get("trending_thresholds") || {};
        trendingEnabled.checked = th.enabled === true;
        downloadSurge.value = th.download_surge_count || 10000;
        trendingScore.value = th.trending_score || 50;
        timeWindow.value = th.time_window_hours || 24;
        updateThresholdFieldsState(th.enabled === true);
    }

    function updateThresholdFieldsState(enabled) {
        var fields = [downloadSurge, trendingScore, timeWindow];
        for (var i = 0; i < fields.length; i++) {
            fields[i].disabled = !enabled;
            fields[i].closest(".setting-row").classList.toggle("setting-disabled", !enabled);
        }
    }

    function bindThresholdEvents() {
        trendingEnabled.addEventListener("change", function () {
            useSettings.set("trending_thresholds.enabled", trendingEnabled.checked);
            updateThresholdFieldsState(trendingEnabled.checked);
        });

        downloadSurge.addEventListener("input", function () {
            var val = parseInt(downloadSurge.value, 10);
            if (!isNaN(val) && val > 0) {
                useSettings.set("trending_thresholds.download_surge_count", val);
            }
        });

        trendingScore.addEventListener("input", function () {
            var val = parseInt(trendingScore.value, 10);
            if (!isNaN(val) && val >= 0) {
                useSettings.set("trending_thresholds.trending_score", val);
            }
        });

        timeWindow.addEventListener("input", function () {
            var val = parseInt(timeWindow.value, 10);
            if (!isNaN(val) && val >= 1 && val <= 168) {
                useSettings.set("trending_thresholds.time_window_hours", val);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Section: Schema field visibility
    // -----------------------------------------------------------------------

    /** Category display names (for grouping headers). */
    var CATEGORY_LABELS = {
        basic: "Basic Info",
        performance: "Performance Benchmarks",
        practical: "Practical Details",
        deployment: "Deployment",
        community: "Community",
        cost: "Cost & Pricing",
        provider: "Provider",
    };

    /** Category display order. */
    var CATEGORY_ORDER = [
        "basic", "performance", "practical",
        "deployment", "community", "cost", "provider",
    ];

    function populateSchema() {
        var fields = useSettings.get("schema_fields") || {};
        var html = "";

        // Group fields by category
        var grouped = {};
        var fieldKeys = Object.keys(fields);
        for (var i = 0; i < fieldKeys.length; i++) {
            var key = fieldKeys[i];
            var field = fields[key];
            var cat = field.category || "basic";
            if (!grouped[cat]) grouped[cat] = [];
            grouped[cat].push({ key: key, field: field });
        }

        // Render by category order
        for (var ci = 0; ci < CATEGORY_ORDER.length; ci++) {
            var catKey = CATEGORY_ORDER[ci];
            var catFields = grouped[catKey];
            if (!catFields || catFields.length === 0) continue;

            html += '<div class="schema-category">';
            html += '<h3 class="schema-category-title">' +
                    escapeHtml(CATEGORY_LABELS[catKey] || catKey) + '</h3>';
            html += '<div class="schema-category-fields">';

            for (var fi = 0; fi < catFields.length; fi++) {
                var sf = catFields[fi];
                html +=
                    '<div class="schema-field-row">' +
                    '<label class="toggle">' +
                    '<input type="checkbox" class="schema-toggle" data-field="' +
                        escapeHtml(sf.key) + '"' +
                        (sf.field.visible ? " checked" : "") + '>' +
                    '<span class="toggle-slider"></span>' +
                    '</label>' +
                    '<div class="schema-field-info">' +
                    '<span class="schema-field-name">' +
                        escapeHtml(sf.field.display_name || sf.key) + '</span>' +
                    '<span class="schema-field-type">' +
                        escapeHtml(sf.field.type || "string") + '</span>' +
                    '</div>' +
                    "</div>";
            }

            html += "</div></div>";
        }

        schemaFields.innerHTML = html;

        // Bind toggle events
        var toggles = schemaFields.querySelectorAll(".schema-toggle");
        for (var ti = 0; ti < toggles.length; ti++) {
            toggles[ti].addEventListener("change", handleSchemaToggle);
        }
    }

    function handleSchemaToggle(e) {
        var fieldKey = e.currentTarget.getAttribute("data-field");
        var checked = e.currentTarget.checked;
        useSettings.set("schema_fields." + fieldKey + ".visible", checked);
    }

    // -----------------------------------------------------------------------
    // Save bar — sticky footer with save/discard actions
    // -----------------------------------------------------------------------
    function updateSaveBar(dirty) {
        if (dirty) {
            saveBar.classList.add("save-bar-visible");
        } else {
            saveBar.classList.remove("save-bar-visible");
        }
    }

    function bindSaveBar() {
        btnSave.addEventListener("click", function () {
            btnSave.disabled = true;
            btnSave.textContent = "Saving...";

            useSettings.save()
                .then(function (data) {
                    btnSave.disabled = false;
                    btnSave.textContent = "Save changes";

                    if (data.warnings && data.warnings.length > 0) {
                        showToast("Saved with warnings: " + data.warnings.join("; "), "warning");
                    } else {
                        showToast("Settings saved successfully", "success");
                    }

                    // Re-populate to reflect any server-side normalization
                    populateAll();
                })
                .catch(function (err) {
                    btnSave.disabled = false;
                    btnSave.textContent = "Save changes";
                    showToast("Save failed: " + err.message, "error");
                });
        });

        btnDiscard.addEventListener("click", function () {
            useSettings.discard();
            populateAll();
            showToast("Changes discarded", "info");
        });
    }

    // -----------------------------------------------------------------------
    // Toast notifications
    // -----------------------------------------------------------------------
    var _toastTimer = null;

    function showToast(message, type) {
        clearTimeout(_toastTimer);
        toastMessage.textContent = message;
        toast.className = "toast toast-visible toast-" + (type || "info");
        toast.style.display = "";

        _toastTimer = setTimeout(function () {
            toast.classList.remove("toast-visible");
            setTimeout(function () { toast.style.display = "none"; }, 300);
        }, 4000);
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------
    function escapeHtml(str) {
        if (!str) return "";
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(String(str)));
        return div.innerHTML;
    }
})();
