/* ======================================
   NEURAL DIGIT — Interactive Frontend Logic
   ====================================== */

document.addEventListener('DOMContentLoaded', () => {
    // ================================
    // TAB NAVIGATION
    // ================================
    window.switchTab = function(tab) {
        const predictorView = document.getElementById('predictorView');
        const analyticsView = document.getElementById('analyticsView');
        const tabPredictor = document.getElementById('tabPredictor');
        const tabAnalytics = document.getElementById('tabAnalytics');

        if (tab === 'predictor') {
            predictorView.style.display = '';
            analyticsView.style.display = 'none';
            tabPredictor.classList.add('active');
            tabAnalytics.classList.remove('active');
        } else {
            predictorView.style.display = 'none';
            analyticsView.style.display = '';
            tabPredictor.classList.remove('active');
            tabAnalytics.classList.add('active');
            loadAnalytics();
        }
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // ================================
    // PIXEL GRID ANIMATION (Bento)
    // ================================
    const pixelGrid = document.getElementById('pixelGrid');
    if (pixelGrid) {
        for (let i = 0; i < 224; i++) {
            const cell = document.createElement('div');
            cell.className = 'pixel-cell';
            cell.style.backgroundColor = `rgba(138, 43, 226, ${Math.random() * 0.3})`;
            pixelGrid.appendChild(cell);
        }
        setInterval(() => {
            const cells = pixelGrid.querySelectorAll('.pixel-cell');
            cells.forEach(cell => {
                if (Math.random() > 0.85) {
                    const hue = Math.random() > 0.5 ? '138, 43, 226' : '0, 251, 251';
                    cell.style.backgroundColor = `rgba(${hue}, ${Math.random() * 0.5 + 0.1})`;
                }
            });
        }, 300);
    }

    // ================================
    // FILE UPLOAD LOGIC
    // ================================
    const uploadArea = document.getElementById('uploadArea');
    const uploadZone = document.getElementById('uploadZone');
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const uploadContent = document.getElementById('uploadContent');
    const predictBtn = document.getElementById('predictBtn');
    const resultBox = document.getElementById('resultBox');
    const resultText = document.getElementById('resultText');

    let selectedFile = null;

    uploadZone.addEventListener('click', () => imageInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'rgba(138, 43, 226, 0.5)';
        uploadZone.style.background = 'rgba(33, 29, 49, 0.6)';
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = '';
        uploadZone.style.background = '';
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '';
        uploadZone.style.background = '';
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (0 or 1 digit).');
            return;
        }
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            uploadContent.style.display = 'none';
        };
        reader.readAsDataURL(file);

        predictBtn.removeAttribute('disabled');
        resultBox.classList.remove('success');
        resultText.textContent = 'Awaiting neural input...';
    }

    // ================================
    // PREDICTION
    // ================================
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        predictBtn.innerHTML = '<span class="analyzing-spinner"></span> Analyzing...';
        predictBtn.setAttribute('disabled', 'true');
        resultBox.classList.remove('success');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();

            resultText.textContent = `Prediction: ${data.prediction}`;
            resultBox.classList.add('success');

        } catch (error) {
            console.error(error);
            resultText.textContent = 'Error during analysis';
        } finally {
            predictBtn.innerHTML = '<span class="material-symbols-outlined">psychology</span> Analyze Digit';
            predictBtn.removeAttribute('disabled');
        }
    });

    // ================================
    // CHART.JS GLOBAL CONFIG
    // ================================
    Chart.defaults.color = '#988ca0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;

    let scatterChart, donutChart, lossChart, weightChart;

    // ================================
    // LOAD ANALYTICS
    // ================================
    async function loadAnalytics() {
        try {
            const res = await fetch('/api/analytics');
            const data = await res.json();
            renderMetrics(data);
            renderScatter(data.scatter_data);
            renderDonut(data.prediction_distribution);
            renderLoss(data.train_history);
            renderConfusion(data.confusion);
            renderWeights(data.weight_histogram, data.weight_stats);
            renderHistory(data.prediction_history);
            animateMetrics(data);
        } catch (err) {
            console.error('Analytics load failed:', err);
        }
    }

    // ================================
    // RENDER METRICS
    // ================================
    function renderMetrics(data) {
        document.getElementById('accuracyValue').textContent = data.accuracy + '%';
        document.getElementById('totalPredictions').textContent = data.total_predictions.toLocaleString();
        document.getElementById('avgResponse').textContent = data.avg_response_ms + 'ms';
        document.getElementById('totalEpochs').textContent = data.training_epochs.toLocaleString();
    }

    function animateMetrics(data) {
        // Accuracy ring
        const ring = document.getElementById('accuracyRing');
        const circumference = 2 * Math.PI * 40;
        const offset = circumference - (data.accuracy / 100) * circumference;
        setTimeout(() => { ring.style.strokeDashoffset = offset; }, 200);

        // Perf bar
        const perf = document.getElementById('perfFill');
        const perfPercent = Math.min(100, Math.max(0, 100 - (data.avg_response_ms / 100 * 100)));
        setTimeout(() => { perf.style.width = perfPercent + '%'; }, 300);

        // Sparkline
        const sparkContainer = document.getElementById('sparkline');
        sparkContainer.innerHTML = '';
        const heights = [20, 45, 35, 60, 55, 80, 70, 90, 40, 65];
        heights.forEach((h, i) => {
            const bar = document.createElement('div');
            bar.className = 'spark-bar';
            bar.style.height = '0%';
            sparkContainer.appendChild(bar);
            setTimeout(() => { bar.style.height = h + '%'; }, 100 + i * 80);
        });
    }

    // ================================
    // SCATTER CHART
    // ================================
    function renderScatter(scatter) {
        const ctx = document.getElementById('scatterChart');
        if (!ctx) return;

        if (scatterChart) scatterChart.destroy();

        scatterChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Digit 0',
                        data: scatter.class0,
                        backgroundColor: 'rgba(220, 184, 255, 0.6)',
                        borderColor: 'rgba(220, 184, 255, 1)',
                        borderWidth: 1,
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        pointHoverBackgroundColor: '#dcb8ff',
                        pointHoverBorderColor: '#fff',
                        pointHoverBorderWidth: 2,
                    },
                    {
                        label: 'Digit 1',
                        data: scatter.class1,
                        backgroundColor: 'rgba(0, 251, 251, 0.6)',
                        borderColor: 'rgba(0, 251, 251, 1)',
                        borderWidth: 1,
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        pointHoverBackgroundColor: '#00fbfb',
                        pointHoverBorderColor: '#fff',
                        pointHoverBorderWidth: 2,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 1500, easing: 'easeOutQuart' },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: { family: "'Space Grotesk', sans-serif", weight: 'bold' }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 11, 30, 0.9)',
                        borderColor: 'rgba(138, 43, 226, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: "'Space Grotesk'", weight: 'bold' },
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            label: (ctx) => `Decision: ${ctx.parsed.x.toFixed(3)}`
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Decision Function Value',
                            font: { family: "'Space Grotesk'", weight: 'bold', size: 10 },
                            color: '#988ca0'
                        },
                        grid: {
                            color: (ctx) => ctx.tick.value === 0 ? 'rgba(255,176,202,0.4)' : 'rgba(255,255,255,0.05)'
                        },
                        ticks: { font: { size: 9 } }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Spread',
                            font: { family: "'Space Grotesk'", weight: 'bold', size: 10 },
                            color: '#988ca0'
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { font: { size: 9 } }
                    }
                }
            }
        });
    }

    // ================================
    // DONUT CHART
    // ================================
    function renderDonut(dist) {
        const ctx = document.getElementById('donutChart');
        if (!ctx) return;

        if (donutChart) donutChart.destroy();

        const total = dist.digit_0 + dist.digit_1;
        const pct0 = total > 0 ? ((dist.digit_0 / total) * 100).toFixed(1) : 50;
        const pct1 = total > 0 ? ((dist.digit_1 / total) * 100).toFixed(1) : 50;

        donutChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Digit 0', 'Digit 1'],
                datasets: [{
                    data: [dist.digit_0 || 50, dist.digit_1 || 50],
                    backgroundColor: [
                        'rgba(220, 184, 255, 0.8)',
                        'rgba(0, 251, 251, 0.8)'
                    ],
                    borderColor: [
                        'rgba(220, 184, 255, 1)',
                        'rgba(0, 251, 251, 1)'
                    ],
                    borderWidth: 2,
                    hoverOffset: 8,
                    hoverBorderWidth: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '65%',
                animation: { animateRotate: true, duration: 1500 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 11, 30, 0.9)',
                        borderColor: 'rgba(138, 43, 226, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: "'Space Grotesk'", weight: 'bold' },
                        padding: 12,
                    }
                }
            }
        });

        // Legend
        const legend = document.getElementById('donutLegend');
        legend.innerHTML = `
            <div class="legend-item">
                <div class="legend-dot purple"></div>
                <div>
                    <div class="legend-name">Digit '0'</div>
                    <div class="legend-value purple">${pct0}%</div>
                </div>
            </div>
            <div class="legend-item">
                <div class="legend-dot cyan"></div>
                <div>
                    <div class="legend-name">Digit '1'</div>
                    <div class="legend-value cyan">${pct1}%</div>
                </div>
            </div>
        `;
    }

    // ================================
    // LOSS CURVE
    // ================================
    function renderLoss(history) {
        const ctx = document.getElementById('lossChart');
        if (!ctx) return;

        if (lossChart) lossChart.destroy();

        const labels = history.map(h => h.epoch);
        const losses = history.map(h => h.loss);

        const gradient = ctx.getContext('2d').createLinearGradient(0, 0, ctx.width || 600, 0);
        gradient.addColorStop(0, '#8a2be2');
        gradient.addColorStop(1, '#00fbfb');

        const gradientFill = ctx.getContext('2d').createLinearGradient(0, 0, 0, ctx.height || 300);
        gradientFill.addColorStop(0, 'rgba(138, 43, 226, 0.15)');
        gradientFill.addColorStop(1, 'rgba(0, 251, 251, 0)');

        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Loss',
                    data: losses,
                    borderColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    backgroundColor: gradientFill,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#00fbfb',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 2000, easing: 'easeOutQuart' },
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 11, 30, 0.9)',
                        borderColor: 'rgba(138, 43, 226, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: "'Space Grotesk'", weight: 'bold' },
                        padding: 12,
                        callbacks: {
                            title: (items) => `Epoch ${items[0].label}`,
                            label: (ctx) => `Loss: ${ctx.parsed.y.toFixed(4)}`
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch',
                            font: { family: "'Space Grotesk'", weight: 'bold', size: 10 },
                            color: '#988ca0'
                        },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { maxTicksLimit: 10, font: { size: 9 } }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss',
                            font: { family: "'Space Grotesk'", weight: 'bold', size: 10 },
                            color: '#988ca0'
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { font: { size: 9 } }
                    }
                }
            }
        });
    }

    // ================================
    // CONFUSION MATRIX
    // ================================
    function renderConfusion(cm) {
        if (!cm) return;
        document.getElementById('cmTP').textContent = cm.tp;
        document.getElementById('cmFP').textContent = cm.fp;
        document.getElementById('cmFN').textContent = cm.fn;
        document.getElementById('cmTN').textContent = cm.tn;
    }

    // ================================
    // WEIGHT DISTRIBUTION
    // ================================
    function renderWeights(hist, stats) {
        const ctx = document.getElementById('weightChart');
        if (!ctx || !hist) return;

        if (weightChart) weightChart.destroy();

        const labels = hist.edges.slice(0, -1).map((e, i) =>
            ((e + hist.edges[i + 1]) / 2).toFixed(3)
        );

        const maxCount = Math.max(...hist.counts);
        const colors = hist.counts.map(c => {
            const intensity = c / maxCount;
            const r = Math.round(138 + (0 - 138) * intensity);
            const g = Math.round(43 + (251 - 43) * intensity);
            const b = Math.round(226 + (251 - 226) * intensity);
            return `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.5})`;
        });

        const hoverColors = hist.counts.map(c => {
            const intensity = c / maxCount;
            const r = Math.round(138 + (0 - 138) * intensity);
            const g = Math.round(43 + (251 - 43) * intensity);
            const b = Math.round(226 + (251 - 226) * intensity);
            return `rgba(${r}, ${g}, ${b}, ${0.6 + intensity * 0.4})`;
        });

        weightChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Frequency',
                    data: hist.counts,
                    backgroundColor: colors,
                    hoverBackgroundColor: hoverColors,
                    borderColor: colors.map(c => c.replace(/[\d.]+\)$/, '0.8)')),
                    borderWidth: 1,
                    borderRadius: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 1500, easing: 'easeOutQuart' },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 11, 30, 0.9)',
                        borderColor: 'rgba(138, 43, 226, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: "'Space Grotesk'", weight: 'bold' },
                        padding: 12,
                        callbacks: {
                            title: (items) => `Weight: ${items[0].label}`,
                            label: (ctx) => `Count: ${ctx.parsed.y}`
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Weight Value',
                            font: { family: "'Space Grotesk'", weight: 'bold', size: 10 },
                            color: '#988ca0'
                        },
                        grid: { display: false },
                        ticks: { maxTicksLimit: 15, font: { size: 8 }, maxRotation: 0 }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency',
                            font: { family: "'Space Grotesk'", weight: 'bold', size: 10 },
                            color: '#988ca0'
                        },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { font: { size: 9 } }
                    }
                }
            }
        });

        // Stats
        if (stats) {
            document.getElementById('weightStats').innerHTML = `
                <span class="weight-sigma">σ = ${stats.std}</span>
                <p class="weight-sub">Standard Deviation</p>
            `;
        }
    }

    // ================================
    // PREDICTION HISTORY
    // ================================
    function renderHistory(history) {
        const tbody = document.getElementById('historyBody');
        if (!tbody) return;

        if (!history || history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="empty-history">No predictions yet. Upload an image to start.</td></tr>';
            return;
        }

        tbody.innerHTML = history.map(h => `
            <tr>
                <td class="td-id">${h.id}</td>
                <td>
                    <div class="td-input-cell">${h.predicted}</div>
                </td>
                <td class="td-pred">${h.predicted}</td>
                <td class="td-conf">${h.confidence}%</td>
                <td class="text-right td-time">${h.time_ms}ms</td>
            </tr>
        `).join('');
    }
});
