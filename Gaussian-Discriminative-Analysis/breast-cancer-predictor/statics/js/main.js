const state = {
    metadata: null,
    values: {},
    activeGroup: "mean",
    charts: {
        decision: null,
        distribution: null,
        separation: null,
        correlation: null,
    },
    predictTimer: null,
    vizTimer: null,
    analyticsLoaded: false,
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
    try {
        state.metadata = await fetchJson("/api/metadata");
        hydrateHeader();
        setMedianValues();
        buildAxisOptions();
        bindNavbar();
        bindStaticControls();
        renderFeatureControls();
        await updatePrediction();
        await updateVisualization();
    } catch (error) {
        console.error(error);
        document.getElementById("predictionDetail").textContent = "Could not load model metadata.";
    }
}

function hydrateHeader() {
    const accuracy = Math.round(state.metadata.model.training_accuracy * 1000) / 10;
    document.getElementById("modelAccuracy").textContent = `Accuracy ${accuracy}%`;
    document.getElementById("datasetRows").textContent = `${state.metadata.rows} records`;
}

function bindNavbar() {
    document.querySelectorAll(".nav-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            const target = tab.dataset.target;
            document.getElementById("form-section").style.display = target === "form-section" ? "block" : "none";
            document.getElementById("analytics-section").style.display = target === "analytics-section" ? "block" : "none";

            if (target === "analytics-section" && !state.analyticsLoaded) {
                state.analyticsLoaded = true;
                loadAnalytics();
            }
        });
    });
}

async function loadAnalytics() {
    try {
        const [confusion, correlation, classStats] = await Promise.all([
            fetchJson("/api/confusion"),
            fetchJson("/api/correlation"),
            fetchJson("/api/class-stats"),
        ]);
        renderConfusion(confusion);
        renderCorrelation(correlation);
        renderClassStats(classStats);
    } catch (error) {
        console.error("Failed to load analytics:", error);
    }
}

function renderConfusion(data) {
    document.getElementById("confTP").textContent = data.tp;
    document.getElementById("confTN").textContent = data.tn;
    document.getElementById("confFP").textContent = data.fp;
    document.getElementById("confFN").textContent = data.fn;

    document.getElementById("statAccuracy").textContent = `${(data.accuracy * 100).toFixed(1)}%`;
    document.getElementById("statPrecision").textContent = `${(data.precision * 100).toFixed(1)}%`;
    document.getElementById("statRecall").textContent = `${(data.recall * 100).toFixed(1)}%`;
    document.getElementById("statF1").textContent = `${(data.f1 * 100).toFixed(1)}%`;
}

function renderCorrelation(data) {
    const ctx = document.getElementById("correlationChart");
    destroyChart("correlation");

    const points = [];
    const n = data.labels.length;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            points.push({ x: j, y: i, v: data.matrix[i][j] });
        }
    }

    state.charts.correlation = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [{
                data: points,
                pointRadius: (ctx) => {
                    const area = ctx.chart.chartArea;
                    if (!area) return 10;
                    return Math.min((area.width / n) * 0.42, (area.height / n) * 0.42);
                },
                pointStyle: "rect",
                backgroundColor: (ctx) => {
                    const v = ctx.raw?.v ?? 0;
                    if (v > 0) return `rgba(0, 242, 255, ${Math.abs(v) * 0.7 + 0.1})`;
                    return `rgba(255, 0, 85, ${Math.abs(v) * 0.7 + 0.1})`;
                },
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(10, 12, 16, 0.94)",
                    titleColor: "#FFFFFF",
                    bodyColor: "rgba(255, 255, 255, 0.7)",
                    borderColor: "rgba(0, 242, 255, 0.2)",
                    borderWidth: 1,
                    callbacks: {
                        title: (items) => {
                            const raw = items[0]?.raw;
                            if (!raw) return "";
                            return `${data.labels[raw.y]} × ${data.labels[raw.x]}`;
                        },
                        label: (item) => `Correlation: ${item.raw.v.toFixed(3)}`,
                    },
                },
            },
            scales: {
                x: {
                    type: "linear",
                    min: -0.5,
                    max: n - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: (v) => data.labels[v] || "",
                        color: "rgba(255,255,255,0.5)",
                        maxRotation: 45,
                        font: { size: 10 },
                    },
                    grid: { display: false },
                },
                y: {
                    type: "linear",
                    min: -0.5,
                    max: n - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: (v) => data.labels[v] || "",
                        color: "rgba(255,255,255,0.5)",
                        font: { size: 10 },
                    },
                    grid: { display: false },
                    reverse: true,
                },
            },
        },
    });
}

function renderClassStats(data) {
    const tbody = document.querySelector("#classStatsTable tbody");
    const top15 = data.features.slice(0, 15);
    tbody.innerHTML = top15.map((f) => {
        let deltaClass = "delta-low";
        if (f.diff_pct > 100) deltaClass = "delta-high";
        else if (f.diff_pct > 40) deltaClass = "delta-med";
        return `<tr>
            <td>${f.label}</td>
            <td>${f.benign_mean.toFixed(3)} ± ${f.benign_std.toFixed(3)}</td>
            <td>${f.malignant_mean.toFixed(3)} ± ${f.malignant_std.toFixed(3)}</td>
            <td class="${deltaClass}">${f.diff_pct.toFixed(1)}%</td>
        </tr>`;
    }).join("");
}

function bindStaticControls() {
    document.querySelectorAll(".tab").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
            button.classList.add("active");
            state.activeGroup = button.dataset.group;
            renderFeatureControls();
        });
    });

    document.getElementById("randomBtn").addEventListener("click", loadRandomSample);
    document.getElementById("resetBtn").addEventListener("click", () => {
        setMedianValues();
        document.getElementById("sampleLabel").textContent = "Median profile";
        renderFeatureControls();
        schedulePrediction();
        scheduleVisualization();
    });

    document.getElementById("xFeature").addEventListener("change", scheduleVisualization);
    document.getElementById("yFeature").addEventListener("change", scheduleVisualization);
}

function setMedianValues() {
    state.metadata.features.forEach((feature) => {
        state.values[feature.name] = feature.median;
    });
}

function buildAxisOptions() {
    const xSelect = document.getElementById("xFeature");
    const ySelect = document.getElementById("yFeature");
    const options = state.metadata.features
        .map((feature) => `<option value="${feature.name}">${feature.label}</option>`)
        .join("");

    xSelect.innerHTML = options;
    ySelect.innerHTML = options;
    xSelect.value = "radius_mean";
    ySelect.value = "texture_mean";
}

function renderFeatureControls() {
    const container = document.getElementById("featureControls");
    const groupFeatures = state.metadata.features.filter((feature) => featureGroup(feature.name) === state.activeGroup);
    container.innerHTML = groupFeatures.map(featureControlTemplate).join("");

    container.querySelectorAll(".feature-control").forEach((control) => {
        const name = control.dataset.feature;
        const range = control.querySelector("input[type='range']");
        const number = control.querySelector("input[type='number']");
        const feature = featureByName(name);

        const sync = (value) => {
            const numeric = clamp(parseFloat(value), feature.min, feature.max);
            state.values[name] = numeric;
            range.value = numeric;
            number.value = formatInputValue(numeric);
            updateInfo(feature);
            schedulePrediction();
            scheduleVisualization();
        };

        range.addEventListener("input", (event) => sync(event.target.value));
        number.addEventListener("input", (event) => sync(event.target.value));
        range.addEventListener("focus", () => updateInfo(feature));
        number.addEventListener("focus", () => updateInfo(feature));
        control.querySelector(".info-button").addEventListener("click", () => updateInfo(feature));
    });

    if (groupFeatures[0]) updateInfo(groupFeatures[0]);
}

function featureControlTemplate(feature) {
    const value = state.values[feature.name] ?? feature.median;
    const step = sliderStep(feature);
    return `
        <div class="feature-control" data-feature="${feature.name}">
            <div>
                <label for="${feature.name}">
                    ${feature.label}
                    <button class="info-button" type="button" aria-label="Explain ${feature.label}">
                        <span class="material-symbols-outlined">help</span>
                    </button>
                </label>
                <input id="${feature.name}" type="range" min="${feature.min}" max="${feature.max}" step="${step}" value="${value}">
            </div>
            <input type="number" min="${feature.min}" max="${feature.max}" step="${step}" value="${formatInputValue(value)}" aria-label="${feature.label} value">
        </div>
    `;
}

function updateInfo(feature) {
    document.getElementById("infoTitle").textContent = feature.label;
    document.getElementById("infoDescription").textContent = feature.description;
}

async function loadRandomSample() {
    const button = document.getElementById("randomBtn");
    const original = button.innerHTML;
    button.innerHTML = `<span class="material-symbols-outlined">progress_activity</span> Loading`;
    try {
        const sample = await fetchJson("/api/random");
        state.values = { ...state.values, ...sample.features };
        document.getElementById("sampleLabel").textContent = `${sample.label} #${sample.id}`;
        renderFeatureControls();
        await updatePrediction();
        await updateVisualization();
    } catch (error) {
        console.error(error);
    } finally {
        button.innerHTML = original;
    }
}

function schedulePrediction() {
    clearTimeout(state.predictTimer);
    state.predictTimer = setTimeout(updatePrediction, 160);
}

function scheduleVisualization() {
    clearTimeout(state.vizTimer);
    state.vizTimer = setTimeout(updateVisualization, 260);
}

async function updatePrediction() {
    const result = await postJson("/api/predict", { features: state.values });
    const malignant = result.malignant_probability;
    const percent = Math.round(malignant * 1000) / 10;
    const isMalignant = result.prediction === 1;
    const color = isMalignant ? "#FF0055" : "#00F2FF";
    const detail = isMalignant
        ? "GDA scores this profile closer to the malignant class distribution."
        : "GDA scores this profile closer to the benign class distribution.";

    document.getElementById("riskPercent").textContent = `${percent}%`;
    document.getElementById("predictionLabel").textContent = result.label;
    document.getElementById("predictionLabel").style.color = color;
    document.getElementById("predictionDetail").textContent = detail;
    document.getElementById("riskRing").style.background =
        `conic-gradient(${color} ${malignant * 360}deg, rgba(255,255,255,0.06) 0deg)`;
}

async function updateVisualization() {
    const xFeature = document.getElementById("xFeature").value;
    const yFeature = document.getElementById("yFeature").value;
    const payload = {
        features: state.values,
        x_feature: xFeature,
        y_feature: yFeature,
        grid_size: window.innerWidth < 760 ? 20 : 30,
    };
    const viz = await postJson("/api/visualization", payload);
    drawDecisionChart(viz);
    drawDistributionChart(viz.distribution);
    drawSeparationChart(viz.separation);
}

/* ═══════════ ELLIPSE DRAWING PLUGIN ═══════════ */
const ellipsePlugin = {
    id: "ellipseContours",
    afterDatasetsDraw(chart) {
        const ellipseData = chart.config._ellipseData;
        if (!ellipseData) return;

        const ctx = chart.ctx;
        const xScale = chart.scales.x;
        const yScale = chart.scales.y;

        const classes = [
            { key: "benign", color: "rgba(0, 242, 255, 0.6)", colorFill: "rgba(0, 242, 255, 0.04)" },
            { key: "malignant", color: "rgba(255, 0, 85, 0.6)", colorFill: "rgba(255, 0, 85, 0.04)" },
        ];

        for (const cls of classes) {
            const data = ellipseData[cls.key];
            if (!data) continue;

            // Draw class mean marker
            const mx = xScale.getPixelForValue(data.mean.x);
            const my = yScale.getPixelForValue(data.mean.y);
            ctx.save();
            ctx.beginPath();
            ctx.arc(mx, my, 5, 0, Math.PI * 2);
            ctx.fillStyle = cls.color;
            ctx.fill();
            ctx.strokeStyle = cls.color;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();

            // Draw contour ellipses
            for (const contour of data.contours) {
                const points = contour.points;
                if (points.length < 2) continue;

                ctx.save();
                ctx.beginPath();
                const firstPx = xScale.getPixelForValue(points[0].x);
                const firstPy = yScale.getPixelForValue(points[0].y);
                ctx.moveTo(firstPx, firstPy);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(
                        xScale.getPixelForValue(points[i].x),
                        yScale.getPixelForValue(points[i].y)
                    );
                }
                ctx.closePath();

                if (contour.std === 1) {
                    ctx.fillStyle = cls.colorFill;
                    ctx.fill();
                }

                ctx.strokeStyle = cls.color;
                ctx.lineWidth = contour.std === 1 ? 2 : 1;
                ctx.setLineDash(contour.std === 1 ? [] : [6, 4]);
                ctx.stroke();
                ctx.restore();
            }
        }
    },
};
Chart.register(ellipsePlugin);

function drawDecisionChart(viz) {
    const ctx = document.getElementById("decisionChart");
    destroyChart("decision");

    const surfacePoints = [];
    viz.y_values.forEach((y, rowIndex) => {
        viz.x_values.forEach((x, columnIndex) => {
            const probability = viz.surface[rowIndex][columnIndex];
            surfacePoints.push({ x, y, probability });
        });
    });

    const benign = viz.scatter
        .filter((p) => p.diagnosis === "B")
        .map((p) => ({ x: p.x, y: p.y }));
    const malignant = viz.scatter
        .filter((p) => p.diagnosis === "M")
        .map((p) => ({ x: p.x, y: p.y }));

    const chartInstance = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Probability field",
                    data: surfacePoints,
                    parsing: false,
                    pointRadius: 6,
                    pointHoverRadius: 6,
                    pointStyle: "rect",
                    backgroundColor: (context) => probabilityColor(context.raw?.probability ?? 0),
                    borderWidth: 0,
                },
                {
                    label: "Benign",
                    data: benign,
                    backgroundColor: "rgba(0, 242, 255, 0.6)",
                    borderColor: "rgba(0, 242, 255, 0.9)",
                    pointRadius: 3.5,
                    pointHoverRadius: 7,
                    borderWidth: 1,
                },
                {
                    label: "Malignant",
                    data: malignant,
                    backgroundColor: "rgba(255, 0, 85, 0.6)",
                    borderColor: "rgba(255, 0, 85, 0.9)",
                    pointRadius: 3.5,
                    pointHoverRadius: 7,
                    borderWidth: 1,
                },
                {
                    label: "Current",
                    data: [viz.selected_point],
                    backgroundColor: "#0A0C10",
                    borderColor: "#FFFFFF",
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10,
                },
            ],
        },
        options: chartOptions({
            xTitle: labelFor(viz.x_feature),
            yTitle: labelFor(viz.y_feature),
            legend: true,
            tooltipLabel: (context) => {
                if (context.datasetIndex === 0) {
                    return `P(malignant) = ${(context.raw.probability * 100).toFixed(1)}%`;
                }
                return `${context.dataset.label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
            },
        }),
    });

    // Attach ellipse data for the plugin
    chartInstance.config._ellipseData = viz.ellipses;
    chartInstance.update();
    state.charts.decision = chartInstance;
}

function drawDistributionChart(distribution) {
    const ctx = document.getElementById("distributionChart");
    destroyChart("distribution");
    document.getElementById("distributionTitle").textContent = labelFor(distribution.feature);

    state.charts.distribution = new Chart(ctx, {
        type: "bar",
        data: {
            labels: distribution.labels,
            datasets: [
                {
                    label: "Benign",
                    data: distribution.benign,
                    backgroundColor: "rgba(0, 242, 255, 0.6)",
                    borderRadius: 3,
                    borderSkipped: false,
                },
                {
                    label: "Malignant",
                    data: distribution.malignant,
                    backgroundColor: "rgba(255, 0, 85, 0.55)",
                    borderRadius: 3,
                    borderSkipped: false,
                },
            ],
        },
        options: barOptions("Count"),
    });
}

function drawSeparationChart(separation) {
    const ctx = document.getElementById("separationChart");
    destroyChart("separation");

    state.charts.separation = new Chart(ctx, {
        type: "bar",
        data: {
            labels: separation.map((item) => item.label),
            datasets: [
                {
                    label: "Mean distance",
                    data: separation.map((item) => item.score),
                    backgroundColor: separation.map((_, index) =>
                        index < 4 ? "rgba(191, 0, 255, 0.65)" : "rgba(0, 242, 255, 0.5)"
                    ),
                    borderRadius: 3,
                    borderSkipped: false,
                },
            ],
        },
        options: {
            ...barOptions("Separation"),
            indexAxis: "y",
        },
    });
}

/* ═══════════ CHART OPTION HELPERS ═══════════ */
function chartOptions({ xTitle, yTitle, legend, tooltipLabel }) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                display: legend,
                labels: {
                    usePointStyle: true,
                    boxWidth: 8,
                    padding: 16,
                    color: "rgba(255, 255, 255, 0.55)",
                    font: { size: 11 },
                },
            },
            tooltip: {
                backgroundColor: "rgba(10, 12, 16, 0.92)",
                titleColor: "#FFF",
                bodyColor: "rgba(255,255,255,0.7)",
                borderColor: "rgba(0, 242, 255, 0.2)",
                borderWidth: 1,
                padding: 10,
                callbacks: { label: tooltipLabel },
            },
        },
        scales: {
            x: {
                title: { display: true, text: xTitle, color: "rgba(255,255,255,0.5)", font: { size: 11 } },
                ticks: { color: "rgba(255,255,255,0.45)", font: { size: 10 } },
                grid: { color: "rgba(255,255,255,0.06)" },
            },
            y: {
                title: { display: true, text: yTitle, color: "rgba(255,255,255,0.5)", font: { size: 11 } },
                ticks: { color: "rgba(255,255,255,0.45)", font: { size: 10 } },
                grid: { color: "rgba(255,255,255,0.06)" },
            },
        },
    };
}

function barOptions(yTitle) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    usePointStyle: true,
                    boxWidth: 8,
                    padding: 16,
                    color: "rgba(255,255,255,0.55)",
                    font: { size: 11 },
                },
            },
            tooltip: {
                backgroundColor: "rgba(10, 12, 16, 0.92)",
                titleColor: "#FFF",
                bodyColor: "rgba(255,255,255,0.7)",
                borderColor: "rgba(0, 242, 255, 0.2)",
                borderWidth: 1,
            },
        },
        scales: {
            x: {
                ticks: { maxRotation: 45, minRotation: 0, color: "rgba(255,255,255,0.45)", font: { size: 10 } },
                grid: { display: false },
            },
            y: {
                title: { display: true, text: yTitle, color: "rgba(255,255,255,0.5)", font: { size: 11 } },
                ticks: { color: "rgba(255,255,255,0.45)", font: { size: 10 } },
                grid: { color: "rgba(255,255,255,0.06)" },
            },
        },
    };
}

/* ═══════════ UTILITIES ═══════════ */
function destroyChart(name) {
    if (state.charts[name]) {
        state.charts[name].destroy();
        state.charts[name] = null;
    }
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} failed with ${response.status}`);
    return response.json();
}

async function postJson(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`${url} failed with ${response.status}`);
    return response.json();
}

function featureByName(name) {
    return state.metadata.features.find((feature) => feature.name === name);
}

function labelFor(name) {
    return featureByName(name)?.label ?? name;
}

function featureGroup(name) {
    if (name.endsWith("_se")) return "se";
    if (name.endsWith("_worst")) return "worst";
    return "mean";
}

function sliderStep(feature) {
    const spread = feature.max - feature.min;
    if (spread > 100) return 0.1;
    if (spread > 10) return 0.01;
    return 0.0001;
}

function formatInputValue(value) {
    const abs = Math.abs(value);
    if (abs >= 100) return value.toFixed(1);
    if (abs >= 10) return value.toFixed(2);
    return value.toFixed(4);
}

function clamp(value, min, max) {
    if (Number.isNaN(value)) return min;
    return Math.min(Math.max(value, min), max);
}

function probabilityColor(probability) {
    const alpha = 0.08 + Math.abs(probability - 0.5) * 0.32;
    if (probability >= 0.5) return `rgba(255, 0, 85, ${alpha})`;
    return `rgba(0, 242, 255, ${alpha})`;
}

/* ═══════════ FULLSCREEN CHART MODAL ═══════════ */
let modalChart = null;

function initChartExpandButtons() {
    document.querySelectorAll(".chart-wrap").forEach((wrap) => {
        const canvas = wrap.querySelector("canvas");
        if (!canvas) return;

        const btn = document.createElement("button");
        btn.className = "chart-expand-btn";
        btn.type = "button";
        btn.title = "Expand to fullscreen";
        btn.innerHTML = '<span class="material-symbols-outlined">open_in_full</span>';
        btn.addEventListener("click", () => openChartModal(canvas.id));
        wrap.appendChild(btn);
    });
}

function getChartTitle(canvasId) {
    const titles = {
        decisionChart: "Decision Surface & Gaussian Contours",
        distributionChart: document.getElementById("distributionTitle")?.textContent || "Distribution",
        separationChart: "Top Discriminative Features",
        correlationChart: "Correlation Heatmap",
    };
    return titles[canvasId] || "Chart";
}

function openChartModal(canvasId) {
    const sourceChart = Object.values(state.charts).find(
        (c) => c && c.canvas && c.canvas.id === canvasId
    );
    if (!sourceChart) return;

    const modal = document.getElementById("chartModal");
    const modalCanvas = document.getElementById("chartModalCanvas");
    const title = document.getElementById("chartModalTitle");

    title.textContent = getChartTitle(canvasId);

    // Destroy any previous modal chart
    if (modalChart) {
        modalChart.destroy();
        modalChart = null;
    }

    // Deep clone the config
    const config = sourceChart.config;
    const clonedData = JSON.parse(JSON.stringify(config.data));
    const clonedOptions = JSON.parse(JSON.stringify(config.options));

    // Restore non-serializable callbacks
    clonedOptions.animation = { duration: 300 };

    // Restore backgroundColor functions for scatter datasets
    if (config.type === "scatter" || canvasId === "decisionChart") {
        clonedData.datasets.forEach((ds, i) => {
            const original = config.data.datasets[i];
            if (typeof original.backgroundColor === "function") {
                ds.backgroundColor = original.backgroundColor;
            }
            if (typeof original.pointRadius === "function") {
                ds.pointRadius = original.pointRadius;
            }
        });
    }

    modalChart = new Chart(modalCanvas, {
        type: config.type,
        data: clonedData,
        options: clonedOptions,
    });

    // If it's the decision chart, copy ellipse data
    if (canvasId === "decisionChart" && config._ellipseData) {
        modalChart.config._ellipseData = config._ellipseData;
        modalChart.update();
    }

    modal.classList.add("open");
    document.body.style.overflow = "hidden";
}

function closeChartModal() {
    const modal = document.getElementById("chartModal");
    modal.classList.remove("open");
    document.body.style.overflow = "";
    if (modalChart) {
        modalChart.destroy();
        modalChart = null;
    }
}

// Bind modal close handlers
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("chartModalClose").addEventListener("click", closeChartModal);
    document.getElementById("chartModal").addEventListener("click", (e) => {
        if (e.target === e.currentTarget) closeChartModal();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeChartModal();
    });

    // Inject expand buttons after a short delay to let charts render
    setTimeout(initChartExpandButtons, 1200);
});

// Re-inject buttons when analytics tab opens for the first time
const _origLoadAnalytics = loadAnalytics;
loadAnalytics = async function () {
    await _origLoadAnalytics();
    setTimeout(initChartExpandButtons, 500);
};
