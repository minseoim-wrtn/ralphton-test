/**
 * HF Model Monitor — Dashboard Client
 *
 * Handles table rendering, sorting, filtering, search, and model detail modal.
 * No external dependencies — plain vanilla JS for PM maintainability.
 *
 * Filter state:
 *   - search (text)        — free-text search across name, model_id, author
 *   - category (select)    — exact match on category field
 *   - family (select)      — exact match on author field (model family / org)
 *   - provider (select)    — exact match on provider_name field
 *   - weights (select)     — "open" or "closed" (open_weights boolean)
 *   - paramsMin (number)   — minimum parameter count in billions
 *   - paramsMax (number)   — maximum parameter count in billions
 */

(function () {
    "use strict";

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var allModels = [];       // Full dataset from server
    var displayedModels = []; // After filtering
    var currentSort = { key: "release_date", order: "desc", type: "date" };

    // -----------------------------------------------------------------------
    // Column sort-type map — built once from data-sort-type attributes.
    // Maps column key (data-sort) to its comparator type: "string" | "numeric" | "date"
    // -----------------------------------------------------------------------
    var SORT_TYPE_MAP = {};

    // DOM refs — existing
    var tbody = document.getElementById("model-tbody");
    var searchInput = document.getElementById("search-input");
    var filterCategory = document.getElementById("filter-category");
    var filterWeights = document.getElementById("filter-weights");
    var btnReset = document.getElementById("btn-reset");
    var btnClearSearch = document.getElementById("btn-clear-search");
    var resultCount = document.getElementById("result-count");
    var emptyState = document.getElementById("empty-state");
    var tableWrapper = document.querySelector(".table-wrapper");

    // DOM refs — new filter controls
    var filterFamily = document.getElementById("filter-family");
    var filterProvider = document.getElementById("filter-provider");
    var filterParamsMin = document.getElementById("filter-params-min");
    var filterParamsMax = document.getElementById("filter-params-max");
    var activeFilters = document.getElementById("active-filters");

    // Modal refs
    var modalOverlay = document.getElementById("modal-overlay");
    var modalTitle = document.getElementById("modal-title");
    var modalBody = document.getElementById("modal-body");
    var modalClose = document.getElementById("modal-close");

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    document.addEventListener("DOMContentLoaded", function () {
        buildSortTypeMap();
        loadStats();
        loadCategories();
        loadModels();
        bindEvents();
    });

    /**
     * Build SORT_TYPE_MAP from data-sort-type attributes in the table header.
     * This lets the JS sort logic know whether each column should use
     * string, numeric, or date comparison — without hardcoding a list.
     */
    function buildSortTypeMap() {
        var ths = document.querySelectorAll(".model-table th.sortable");
        for (var i = 0; i < ths.length; i++) {
            var key = ths[i].getAttribute("data-sort");
            var sortType = ths[i].getAttribute("data-sort-type") || "string";
            if (key) {
                SORT_TYPE_MAP[key] = sortType;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Data loading
    // -----------------------------------------------------------------------
    function loadModels() {
        fetch("/api/models")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                allModels = data.models || [];
                populateFilterDropdowns(allModels);
                applyFiltersAndRender();
            })
            .catch(function (err) {
                console.error("Failed to load models:", err);
                tbody.innerHTML =
                    '<tr><td colspan="15" style="text-align:center;padding:40px;color:#c92a2a;">' +
                    "Failed to load models. Is the server running?" +
                    "</td></tr>";
            });
    }

    function loadStats() {
        fetch("/api/stats")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                document.getElementById("stat-total-count").textContent =
                    data.seed_model_count || 0;
                document.getElementById("stat-cat-count").textContent =
                    data.category_count || 0;
                var poll = data.last_poll;
                document.getElementById("stat-last-poll").textContent =
                    poll ? formatDate(poll) : "Never";
            })
            .catch(function () { /* stats are non-critical */ });
    }

    function loadCategories() {
        fetch("/api/categories")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                var cats = data.categories || [];
                cats.forEach(function (cat) {
                    var opt = document.createElement("option");
                    opt.value = cat;
                    opt.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
                    filterCategory.appendChild(opt);
                });
            })
            .catch(function () { /* non-critical */ });
    }

    // -----------------------------------------------------------------------
    // Dropdown population — derived from loaded model data
    // -----------------------------------------------------------------------

    /**
     * Populate the Model Family and Provider dropdowns from the loaded data.
     * Called once after allModels is set. Extracts unique values, sorts them,
     * and builds <option> elements.
     */
    function populateFilterDropdowns(models) {
        var families = {};   // author -> display label
        var providers = {};  // provider_name -> display label

        for (var i = 0; i < models.length; i++) {
            var m = models[i];

            // Model family = author / org
            var author = (m.author || "").trim();
            if (author && !families[author]) {
                families[author] = author;
            }

            // Provider name
            var provName = (m.provider_name || "").trim();
            if (provName && provName !== "N/A" && !providers[provName]) {
                providers[provName] = provName;
            }
        }

        // Populate Model Family dropdown
        var familyKeys = Object.keys(families).sort(function (a, b) {
            return a.toLowerCase().localeCompare(b.toLowerCase());
        });
        for (var fi = 0; fi < familyKeys.length; fi++) {
            var fOpt = document.createElement("option");
            fOpt.value = familyKeys[fi];
            fOpt.textContent = familyKeys[fi];
            filterFamily.appendChild(fOpt);
        }

        // Populate Provider dropdown
        var provKeys = Object.keys(providers).sort(function (a, b) {
            return a.toLowerCase().localeCompare(b.toLowerCase());
        });
        for (var pi = 0; pi < provKeys.length; pi++) {
            var pOpt = document.createElement("option");
            pOpt.value = provKeys[pi];
            pOpt.textContent = provKeys[pi];
            filterProvider.appendChild(pOpt);
        }
    }

    // -----------------------------------------------------------------------
    // Parameter parsing — convert "70B", "1.8T", "671B total (37B active)"
    // into a number in billions for range filtering.
    // -----------------------------------------------------------------------

    /**
     * Parse a params string into a number in billions.
     * Returns null if the value is N/A or unparseable.
     *
     * Examples:
     *   "70B"           -> 70
     *   "1.8T"          -> 1800
     *   "~1.8T (est)"   -> 1800
     *   "671B total (37B active)" -> 671
     *   "128K"          -> 0.000128  (unlikely for params, but handled)
     *   "809M"          -> 0.809
     *   "N/A"           -> null
     */
    function parseParamsBillions(val) {
        if (typeof val === "number") return val / 1e9;
        if (typeof val !== "string") return null;

        var s = val.trim();
        if (!s || s === "N/A") return null;

        // Strip common prefixes
        s = s.replace(/^~/, "").replace(/,/g, "").trim();

        // Match the first number + optional suffix
        var match = s.match(/^([0-9]*\.?[0-9]+)\s*([TBMK])?/);
        if (!match) return null;

        var num = parseFloat(match[1]);
        if (isNaN(num)) return null;

        var suffix = match[2];
        if (suffix === "T") return num * 1000;       // trillion -> billions
        if (suffix === "B") return num;               // already billions
        if (suffix === "M") return num / 1000;        // millions -> billions
        if (suffix === "K") return num / 1000000;     // thousands -> billions

        // No suffix — assume raw number, likely already in billions context
        return num;
    }

    // -----------------------------------------------------------------------
    // Filter state — read current values from all controls
    // -----------------------------------------------------------------------

    /**
     * Read the current filter state from all UI controls.
     * Returns a plain object describing every active filter.
     */
    function getFilterState() {
        var paramsMinVal = filterParamsMin.value.trim();
        var paramsMaxVal = filterParamsMax.value.trim();

        return {
            search: searchInput.value.trim().toLowerCase(),
            category: filterCategory.value,
            family: filterFamily.value,
            provider: filterProvider.value,
            weights: filterWeights.value,
            paramsMin: paramsMinVal !== "" ? parseFloat(paramsMinVal) : null,
            paramsMax: paramsMaxVal !== "" ? parseFloat(paramsMaxVal) : null,
        };
    }

    /**
     * Count how many filters are active (non-empty).
     */
    function countActiveFilters(state) {
        var count = 0;
        if (state.search) count++;
        if (state.category) count++;
        if (state.family) count++;
        if (state.provider) count++;
        if (state.weights) count++;
        if (state.paramsMin !== null) count++;
        if (state.paramsMax !== null) count++;
        return count;
    }

    // -----------------------------------------------------------------------
    // Filtering & sorting (client-side for responsiveness)
    // -----------------------------------------------------------------------
    function applyFiltersAndRender() {
        var state = getFilterState();

        displayedModels = allModels.filter(function (m) {
            // Search filter
            if (state.search) {
                var haystack = (
                    (m.name || "") + " " +
                    (m.model_id || "") + " " +
                    (m.author || "")
                ).toLowerCase();
                if (haystack.indexOf(state.search) === -1) return false;
            }
            // Category filter
            if (state.category && m.category !== state.category) return false;
            // Model family filter (matches on author)
            if (state.family && m.author !== state.family) return false;
            // Provider filter
            if (state.provider && m.provider_name !== state.provider) return false;
            // Weights filter
            if (state.weights === "open" && !m.open_weights) return false;
            if (state.weights === "closed" && m.open_weights) return false;
            // Parameter range filter
            if (state.paramsMin !== null || state.paramsMax !== null) {
                var paramsBillions = parseParamsBillions(m.params);
                // If params is N/A/unknown, exclude from range filter results
                if (paramsBillions === null) return false;
                if (state.paramsMin !== null && paramsBillions < state.paramsMin) return false;
                if (state.paramsMax !== null && paramsBillions > state.paramsMax) return false;
            }
            return true;
        });

        // Sort
        sortModels(displayedModels, currentSort.key, currentSort.order);

        // Render
        renderTable(displayedModels);
        updateResultCount(displayedModels.length, allModels.length);
        updateActiveFiltersBadge(state);
    }

    // -----------------------------------------------------------------------
    // Multi-type sorting — dispatches to string, numeric, or date comparator
    // based on the column's data-sort-type attribute (via SORT_TYPE_MAP).
    //
    // N/A handling: N/A / null / empty values ALWAYS sort to the bottom,
    // regardless of ascending or descending order.
    // -----------------------------------------------------------------------

    /**
     * Sort models array in-place by the given key and order.
     * Uses the column's declared sort type for proper comparison.
     *
     * @param {Array}  models - array of model objects to sort
     * @param {string} key    - column key (matches data-sort attribute)
     * @param {string} order  - "asc" or "desc"
     */
    function sortModels(models, key, order) {
        var sortType = SORT_TYPE_MAP[key] || "string";
        var isDesc = order === "desc";

        models.sort(function (a, b) {
            var va = a[key];
            var vb = b[key];

            var aIsNA = isNAValue(va);
            var bIsNA = isNAValue(vb);

            // N/A always sorts to bottom regardless of direction
            if (aIsNA && bIsNA) return 0;
            if (aIsNA) return 1;
            if (bIsNA) return -1;

            var result;
            switch (sortType) {
                case "numeric":
                    result = compareNumeric(va, vb);
                    break;
                case "date":
                    result = compareDate(va, vb);
                    break;
                default: // "string"
                    result = compareString(va, vb);
                    break;
            }

            return isDesc ? -result : result;
        });
    }

    /**
     * Check if a value should be treated as N/A (missing data).
     * Returns true for null, undefined, empty string, "N/A", or 0 for
     * fields that use 0 as a missing sentinel (downloads/likes excluded).
     */
    function isNAValue(val) {
        if (val === undefined || val === null || val === "") return true;
        if (typeof val === "string") {
            var s = val.trim();
            return s === "" || s === "N/A" || s === "n/a";
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Type-specific comparators
    // -----------------------------------------------------------------------

    /**
     * Numeric comparator — handles raw numbers, booleans, and strings
     * with numeric content like "$2.50", "70B", "128K", "~1.8T".
     * Returns negative if a < b, positive if a > b, 0 if equal.
     */
    function compareNumeric(a, b) {
        var na = parseNumeric(a);
        var nb = parseNumeric(b);

        // Both parsed successfully — straightforward comparison
        if (na !== null && nb !== null) return na - nb;

        // One parsed, one didn't — parsed value comes first
        if (na !== null) return 1;
        if (nb !== null) return -1;

        // Neither parsed — fall back to string comparison
        return compareString(a, b);
    }

    /**
     * Date comparator — handles ISO date strings (YYYY-MM-DD),
     * full ISO timestamps, and falls back to string comparison.
     * Returns negative if a < b, positive if a > b, 0 if equal.
     */
    function compareDate(a, b) {
        var da = parseDate(a);
        var db = parseDate(b);

        if (da !== null && db !== null) return da - db;
        if (da !== null) return 1;
        if (db !== null) return -1;

        return compareString(a, b);
    }

    /**
     * String comparator — case-insensitive locale-aware comparison.
     * Returns negative if a < b, positive if a > b, 0 if equal.
     */
    function compareString(a, b) {
        var sa = String(a || "").toLowerCase();
        var sb = String(b || "").toLowerCase();
        return sa < sb ? -1 : sa > sb ? 1 : 0;
    }

    // -----------------------------------------------------------------------
    // Value parsers for comparators
    // -----------------------------------------------------------------------

    /**
     * Parse a value into a number for numeric comparison.
     * Handles: raw numbers, booleans, "$2.50", "70B", "~1.8T", "128K",
     * "809M", comma-separated numbers ("1,234"), percentages.
     * Returns null if the value cannot be parsed.
     */
    function parseNumeric(val) {
        if (typeof val === "number") return val;
        if (typeof val === "boolean") return val ? 1 : 0;
        if (typeof val !== "string") return null;

        var s = val.trim();
        if (!s || s === "N/A" || s === "n/a") return null;

        // Strip currency, approximate, and thousands separators
        s = s.replace(/^\$/, "").replace(/^~/, "").replace(/,/g, "").trim();

        // Handle suffixes: T (trillion), B (billion), M (million), K (thousand)
        var multipliers = { T: 1e12, B: 1e9, M: 1e6, K: 1e3 };
        var match = s.match(/^([0-9]*\.?[0-9]+)\s*([TBMK])?/);
        if (match) {
            var num = parseFloat(match[1]);
            if (isNaN(num)) return null;
            if (match[2] && multipliers[match[2]]) {
                num *= multipliers[match[2]];
            }
            return num;
        }

        return null;
    }

    /**
     * Parse a value into a timestamp (ms since epoch) for date comparison.
     * Handles: "YYYY-MM-DD", full ISO 8601, and Date-parseable strings.
     * Returns null if the value cannot be parsed as a date.
     */
    function parseDate(val) {
        if (typeof val !== "string") return null;

        var s = val.trim();
        if (!s || s === "N/A" || s === "n/a") return null;

        // Quick sanity check: starts with 4-digit year
        if (!/^\d{4}/.test(s)) return null;

        var ts = new Date(s).getTime();
        return isNaN(ts) ? null : ts;
    }

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------
    function renderTable(models) {
        if (models.length === 0) {
            tbody.innerHTML = "";
            tableWrapper.style.display = "none";
            emptyState.style.display = "block";
            return;
        }

        tableWrapper.style.display = "";
        emptyState.style.display = "none";

        // Build HTML rows in batch (faster than individual DOM ops)
        var html = "";
        for (var i = 0; i < models.length; i++) {
            html += buildRow(models[i]);
        }
        tbody.innerHTML = html;

        // Update sort arrow display
        updateSortArrows();
    }

    function buildRow(m) {
        var catClass = "cat-" + (m.category || "default");
        var catLabel = m.category
            ? m.category.charAt(0).toUpperCase() + m.category.slice(1)
            : "N/A";

        return (
            '<tr data-model-id="' + escapeHtml(m.model_id) + '">' +
            // Name
            '<td><span class="cell-name" onclick="window._openDetail(\'' +
            escapeHtml(m.model_id) + "')" + '">' +
            escapeHtml(m.name) + "</span></td>" +
            // Org
            '<td class="cell-org">' + escapeHtml(m.author) + "</td>" +
            // Category
            '<td><span class="cell-category ' + catClass + '">' +
            catLabel + "</span></td>" +
            // Params
            "<td>" + renderValue(m.params) + "</td>" +
            // Release date
            '<td class="cell-date">' + renderDate(m.release_date) + "</td>" +
            // Benchmarks
            "<td>" + renderBenchmark(m.mmlu) + "</td>" +
            "<td>" + renderBenchmark(m.humaneval) + "</td>" +
            "<td>" + renderBenchmark(m.gpqa) + "</td>" +
            "<td>" + renderBenchmark(m.math) + "</td>" +
            // Price
            '<td class="cell-price">' + renderValue(m.api_price_input) + "</td>" +
            '<td class="cell-price">' + renderValue(m.api_price_output) + "</td>" +
            // Context
            "<td>" + renderValue(m.context_window) + "</td>" +
            // License
            '<td class="cell-license" title="' + escapeHtml(m.license || "N/A") + '">' +
            renderValue(m.license) + "</td>" +
            // Downloads
            '<td class="cell-downloads">' + renderNumber(m.downloads) + "</td>" +
            // Likes
            '<td class="cell-downloads">' + renderNumber(m.likes) + "</td>" +
            "</tr>"
        );
    }

    function renderValue(val) {
        if (val === undefined || val === null || val === "" || val === "N/A") {
            return '<span class="cell-na">N/A</span>';
        }
        return '<span class="cell-number">' + escapeHtml(String(val)) + "</span>";
    }

    function renderBenchmark(val) {
        if (val === undefined || val === null || val === "" || val === "N/A") {
            return '<span class="cell-na">N/A</span>';
        }
        var num = parseFloat(val);
        var cls = "cell-number";
        if (!isNaN(num)) {
            if (num >= 85) cls += " bench-high";
            else if (num >= 70) cls += " bench-mid";
            else cls += " bench-low";
        }
        return '<span class="' + cls + '">' + escapeHtml(String(val)) + "</span>";
    }

    function renderDate(val) {
        if (!val || val === "N/A") return '<span class="cell-na">N/A</span>';
        return escapeHtml(val);
    }

    function renderNumber(val) {
        if (val === undefined || val === null || val === 0) {
            return '<span class="cell-na">0</span>';
        }
        // Format with commas
        return '<span class="cell-number">' + Number(val).toLocaleString() + "</span>";
    }

    function formatDate(isoStr) {
        try {
            var d = new Date(isoStr);
            return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        } catch (e) {
            return isoStr;
        }
    }

    function updateResultCount(shown, total) {
        if (shown === total) {
            resultCount.textContent = total + " models";
        } else {
            resultCount.textContent = shown + " of " + total + " models";
        }
    }

    /**
     * Show/hide the active filters badge and update its count text.
     */
    function updateActiveFiltersBadge(state) {
        var count = countActiveFilters(state);
        if (count > 0) {
            activeFilters.textContent = count + " filter" + (count > 1 ? "s" : "") + " active";
            activeFilters.classList.add("visible");
        } else {
            activeFilters.classList.remove("visible");
        }
    }

    // -----------------------------------------------------------------------
    // Sort arrows — visual indicators on all sortable columns
    //
    // Active column:   solid arrow (▲ asc / ▼ desc) + highlighted header
    // Inactive columns: subtle bi-directional hint (⇅) on hover via CSS
    // -----------------------------------------------------------------------
    function updateSortArrows() {
        var ths = document.querySelectorAll(".model-table th.sortable");
        for (var i = 0; i < ths.length; i++) {
            var th = ths[i];
            var arrow = th.querySelector(".sort-arrow");
            var key = th.getAttribute("data-sort");

            if (key === currentSort.key) {
                // Active sort column — show direction arrow
                th.classList.add("sort-active");
                th.classList.remove("sort-idle");
                arrow.textContent = currentSort.order === "desc" ? " ▼" : " ▲";
                arrow.setAttribute("title",
                    "Sorted " + (currentSort.order === "desc" ? "descending" : "ascending") +
                    " — click to " + (currentSort.order === "desc" ? "sort ascending" : "sort descending"));
            } else {
                // Inactive column — show subtle idle indicator
                th.classList.remove("sort-active");
                th.classList.add("sort-idle");
                arrow.textContent = " ⇅";
                arrow.setAttribute("title", "Click to sort");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Modal — model detail view
    // -----------------------------------------------------------------------
    window._openDetail = function (modelId) {
        // Find the model in our local data first
        var model = null;
        for (var i = 0; i < allModels.length; i++) {
            if (allModels[i].model_id === modelId) {
                model = allModels[i];
                break;
            }
        }

        if (!model) return;

        modalTitle.textContent = model.name || model.model_id;
        modalBody.innerHTML = buildDetailContent(model);
        modalOverlay.style.display = "flex";
        document.body.style.overflow = "hidden";
    };

    function closeModal() {
        modalOverlay.style.display = "none";
        document.body.style.overflow = "";
    }

    function buildDetailContent(m) {
        var hfLink = m.hf_url
            ? '<a href="' + escapeHtml(m.hf_url) + '" target="_blank" rel="noopener">' +
              escapeHtml(m.hf_url) + "</a>"
            : "N/A";

        return (
            '<div class="detail-grid">' +
            detailItem("Model ID", m.model_id) +
            detailItem("Organization", m.author) +
            detailItem("Category", m.category) +
            detailItem("Release Date", m.release_date) +
            detailItem("Parameters", m.params) +
            detailItem("Architecture", m.architecture) +
            detailItem("License", m.license) +
            detailItem("Open Weights", m.open_weights ? "Yes" : "No") +
            "</div>" +
            '<div class="detail-section">' +
            '<div class="detail-section-title">Benchmarks</div>' +
            '<div class="detail-grid">' +
            detailItem("MMLU", m.mmlu) +
            detailItem("HumanEval", m.humaneval) +
            detailItem("GPQA", m.gpqa) +
            detailItem("MATH", m.math) +
            detailItem("Arena ELO", m.arena_elo) +
            "</div></div>" +
            '<div class="detail-section">' +
            '<div class="detail-section-title">Pricing & Deployment</div>' +
            '<div class="detail-grid">' +
            detailItem("Input Price (per 1M tokens)", m.api_price_input) +
            detailItem("Output Price (per 1M tokens)", m.api_price_output) +
            detailItem("Context Window", m.context_window) +
            detailItem("Output Window", m.output_window) +
            detailItem("VRAM Estimate", m.vram_estimate) +
            detailItem("API Available", m.api_available) +
            "</div></div>" +
            '<div class="detail-section">' +
            '<div class="detail-section-title">Community</div>' +
            '<div class="detail-grid">' +
            detailItem("Downloads", m.downloads ? Number(m.downloads).toLocaleString() : "0") +
            detailItem("Likes", m.likes ? Number(m.likes).toLocaleString() : "0") +
            detailItem("Provider", m.provider_name) +
            detailItem("HuggingFace", hfLink) +
            "</div></div>"
        );
    }

    function detailItem(label, value) {
        var displayVal = value === undefined || value === null || value === "" || value === "N/A"
            ? '<span class="cell-na">N/A</span>'
            : escapeHtml(String(value));

        // Allow HTML content (for links)
        if (typeof value === "string" && value.indexOf("<a ") !== -1) {
            displayVal = value;
        }

        return (
            '<div class="detail-item">' +
            '<span class="detail-label">' + escapeHtml(label) + "</span>" +
            '<span class="detail-value">' + displayVal + "</span>" +
            "</div>"
        );
    }

    // -----------------------------------------------------------------------
    // Event binding
    // -----------------------------------------------------------------------
    function bindEvents() {
        // Search with debounce
        var searchTimer = null;
        searchInput.addEventListener("input", function () {
            clearTimeout(searchTimer);
            searchTimer = setTimeout(applyFiltersAndRender, 200);
        });

        // Dropdown filters — immediate
        filterCategory.addEventListener("change", applyFiltersAndRender);
        filterWeights.addEventListener("change", applyFiltersAndRender);
        filterFamily.addEventListener("change", applyFiltersAndRender);
        filterProvider.addEventListener("change", applyFiltersAndRender);

        // Range inputs — debounced like search
        var rangeTimer = null;
        function onRangeInput() {
            clearTimeout(rangeTimer);
            rangeTimer = setTimeout(applyFiltersAndRender, 300);
        }
        filterParamsMin.addEventListener("input", onRangeInput);
        filterParamsMax.addEventListener("input", onRangeInput);

        // Reset all filters
        btnReset.addEventListener("click", resetAllFilters);
        btnClearSearch.addEventListener("click", resetAllFilters);

        // Column sorting
        var sortableHeaders = document.querySelectorAll(".model-table th.sortable");
        for (var i = 0; i < sortableHeaders.length; i++) {
            sortableHeaders[i].addEventListener("click", handleSortClick);
        }

        // Modal close
        modalClose.addEventListener("click", closeModal);
        modalOverlay.addEventListener("click", function (e) {
            if (e.target === modalOverlay) closeModal();
        });
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") closeModal();
        });
    }

    /**
     * Reset all filter controls to their default (empty) state and re-render.
     */
    function resetAllFilters() {
        searchInput.value = "";
        filterCategory.value = "";
        filterFamily.value = "";
        filterProvider.value = "";
        filterWeights.value = "";
        filterParamsMin.value = "";
        filterParamsMax.value = "";
        currentSort = { key: "release_date", order: "desc", type: "date" };
        applyFiltersAndRender();
    }

    function handleSortClick(e) {
        var th = e.currentTarget;
        var key = th.getAttribute("data-sort");
        if (!key) return;

        if (currentSort.key === key) {
            // Toggle order on same column
            currentSort.order = currentSort.order === "desc" ? "asc" : "desc";
        } else {
            // New column — pick a sensible default order based on type:
            //   numeric/date → desc (newest/biggest first)
            //   string       → asc  (A-Z)
            var sortType = SORT_TYPE_MAP[key] || "string";
            currentSort.key = key;
            currentSort.type = sortType;
            currentSort.order = sortType === "string" ? "asc" : "desc";
        }

        applyFiltersAndRender();
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
