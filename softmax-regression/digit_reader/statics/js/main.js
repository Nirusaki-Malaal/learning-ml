/* ======================================
   NEURAL DIGIT — Softmax Regression Frontend
   ====================================== */

// 10-class color palette
const CLASS_COLORS = [
    { bg: 'rgba(220, 184, 255, 0.6)', border: 'rgba(220, 184, 255, 1)',   hover: '#dcb8ff', name: 'Purple'  },   // 0
    { bg: 'rgba(0, 251, 251, 0.6)',   border: 'rgba(0, 251, 251, 1)',     hover: '#00fbfb', name: 'Cyan'    },   // 1
    { bg: 'rgba(255, 176, 202, 0.6)', border: 'rgba(255, 176, 202, 1)',   hover: '#ffb0ca', name: 'Pink'    },   // 2
    { bg: 'rgba(129, 230, 217, 0.6)', border: 'rgba(129, 230, 217, 1)',   hover: '#81e6d9', name: 'Teal'    },   // 3
    { bg: 'rgba(251, 211, 141, 0.6)', border: 'rgba(251, 211, 141, 1)',   hover: '#fbd38d', name: 'Gold'    },   // 4
    { bg: 'rgba(183, 148, 244, 0.6)', border: 'rgba(183, 148, 244, 1)',   hover: '#b794f4', name: 'Violet'  },   // 5
    { bg: 'rgba(144, 205, 244, 0.6)', border: 'rgba(144, 205, 244, 1)',   hover: '#90cdf4', name: 'Blue'    },   // 6
    { bg: 'rgba(252, 180, 100, 0.6)', border: 'rgba(252, 180, 100, 1)',   hover: '#fcb464', name: 'Orange'  },   // 7
    { bg: 'rgba(104, 211, 145, 0.6)', border: 'rgba(104, 211, 145, 1)',   hover: '#68d391', name: 'Green'   },   // 8
    { bg: 'rgba(252, 129, 129, 0.6)', border: 'rgba(252, 129, 129, 1)',   hover: '#fc8181', name: 'Red'     },   // 9
];

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
                    const colorSets = [
                        '138, 43, 226',
                        '0, 251, 251',
                        '255, 176, 202',
                        '129, 230, 217',
                        '251, 211, 141'
                    ];
                    const hue = colorSets[Math.floor(Math.random() * colorSets.length)];
                    cell.style.backgroundColor = `rgba(${hue}, ${Math.random() * 0.5 + 0.1})`;
                }
            });
        }, 300);
    }

    // ================================
    // FILE UPLOAD LOGIC
    // ================================
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
            alert('Please select a valid image file (digit 0-9).');
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
        document.getElementById('confidencePanel').style.display = 'none';
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

            resultText.textContent = `Prediction: ${data.prediction}  (${data.confidence}%)`;
            resultBox.classList.add('success');

            // Render per-class confidence bars
            renderConfidenceBars(data.probabilities, data.prediction);

        } catch (error) {
            console.error(error);
            resultText.textContent = 'Error during analysis';
        } finally {
            predictBtn.innerHTML = '<span class="material-symbols-outlined">psychology</span> Analyze Digit';
            predictBtn.removeAttribute('disabled');
        }
    });

    function renderConfidenceBars(probabilities, predicted) {
        const panel = document.getElementById('confidencePanel');
        const barsContainer = document.getElementById('confidenceBars');
        panel.style.display = 'block';

        let html = '';
        for (let i = 0; i < 10; i++) {
            const pct = probabilities[String(i)] || 0;
            const isTop = (i === predicted);
            const color = CLASS_COLORS[i];
            html += `
                <div class="conf-bar-row ${isTop ? 'conf-bar-top' : ''}">
                    <span class="conf-digit">${i}</span>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill" style="width:${pct}%; background:${color.border}; transition: width 0.8s ease ${i * 0.05}s"></div>
                    </div>
                    <span class="conf-pct">${pct.toFixed(1)}%</span>
                </div>
            `;
        }
        barsContainer.innerHTML = html;
    }

    // ================================
    // CHART.JS GLOBAL CONFIG
    // ================================
    Chart.defaults.color = '#988ca0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;

    let scatterChart, donutChart, lossChart, weightChart, classAccChart;

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
            renderConfusionMatrix10(data.confusion_matrix);
            renderWeights(data.weight_histogram, data.weight_stats);
            renderHistory(data.prediction_history);
            renderClassAccuracy(data.per_class_accuracy);
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
        const ring = document.getElementById('accuracyRing');
        const circumference = 2 * Math.PI * 40;
        const offset = circumference - (data.accuracy / 100) * circumference;
        setTimeout(() => { ring.style.strokeDashoffset = offset; }, 200);

        const perf = document.getElementById('perfFill');
        const perfPercent = Math.min(100, Math.max(0, 100 - (data.avg_response_ms / 100 * 100)));
        setTimeout(() => { perf.style.width = perfPercent + '%'; }, 300);

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
    // PCA SCATTER CHART (10 classes)
    // ================================
    function renderScatter(scatter) {
        const ctx = document.getElementById('scatterChart');
        if (!ctx) return;
        if (scatterChart) scatterChart.destroy();

        const datasets = [];
        for (let i = 0; i < 10; i++) {
            const key = `class${i}`;
            if (scatter[key] && scatter[key].length > 0) {
                datasets.push({
                    label: `Digit ${i}`,
                    data: scatter[key],
                    backgroundColor: CLASS_COLORS[i].bg,
                    borderColor: CLASS_COLORS[i].border,
                    borderWidth: 1,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: CLASS_COLORS[i].hover,
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                });
            }
        }

        scatterChart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 1500, easing: 'easeOutQuart' },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 12,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: { family: "'Space Grotesk', sans-serif", weight: 'bold', size: 9 }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 11, 30, 0.9)',
                        borderColor: 'rgba(138, 43, 226, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: "'Space Grotesk'", weight: 'bold' },
                        padding: 12,
                        displayColors: true,
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'PCA Component 1', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { font: { size: 9 } }
                    },
                    y: {
                        title: { display: true, text: 'PCA Component 2', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { font: { size: 9 } }
                    }
                }
            }
        });
    }

    // ================================
    // DONUT CHART (10 digits)
    // ================================
    function renderDonut(dist) {
        const ctx = document.getElementById('donutChart');
        if (!ctx) return;
        if (donutChart) donutChart.destroy();

        const labels = [];
        const values = [];
        const bgColors = [];
        const borderColors = [];

        for (let i = 0; i < 10; i++) {
            labels.push(`Digit ${i}`);
            values.push(dist[`digit_${i}`] || 0);
            bgColors.push(CLASS_COLORS[i].bg);
            borderColors.push(CLASS_COLORS[i].border);
        }

        // If no predictions yet, show equal distribution
        const total = values.reduce((a, b) => a + b, 0);
        const displayValues = total === 0 ? values.map(() => 1) : values;

        donutChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: displayValues,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    hoverOffset: 8,
                    hoverBorderWidth: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '55%',
                animation: { animateRotate: true, duration: 1500 },
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            padding: 8,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: { family: "'Space Grotesk', sans-serif", size: 10 }
                        }
                    },
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
        const accuracies = history.map(h => h.accuracy * 100);

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
                datasets: [
                    {
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
                        yAxisID: 'y',
                    },
                    {
                        label: 'Accuracy %',
                        data: accuracies,
                        borderColor: 'rgba(129, 230, 217, 0.7)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        pointHoverBackgroundColor: '#81e6d9',
                        yAxisID: 'y1',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 2000, easing: 'easeOutQuart' },
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { padding: 16, usePointStyle: true, font: { size: 10 } }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 11, 30, 0.9)',
                        borderColor: 'rgba(138, 43, 226, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: "'Space Grotesk'", weight: 'bold' },
                        padding: 12,
                        callbacks: {
                            title: (items) => `Epoch ${items[0].label}`,
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Epoch', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { maxTicksLimit: 10, font: { size: 9 } }
                    },
                    y: {
                        title: { display: true, text: 'Cross-Entropy Loss', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { font: { size: 9 } },
                        position: 'left',
                    },
                    y1: {
                        title: { display: true, text: 'Accuracy %', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#81e6d9' },
                        grid: { display: false },
                        ticks: { font: { size: 9 }, color: '#81e6d9' },
                        position: 'right',
                    }
                }
            }
        });
    }

    // ================================
    // 10×10 CONFUSION MATRIX
    // ================================
    function renderConfusionMatrix10(matrix) {
        const container = document.getElementById('confusionMatrix10');
        if (!container || !matrix) return;

        // Find max value for color scaling
        let maxVal = 0;
        matrix.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));

        let html = '<div class="cm10-grid">';

        // Header row
        html += '<div class="cm10-corner">↓T / P→</div>';
        for (let j = 0; j < 10; j++) {
            html += `<div class="cm10-header">${j}</div>`;
        }

        // Data rows
        for (let i = 0; i < 10; i++) {
            html += `<div class="cm10-row-label">${i}</div>`;
            for (let j = 0; j < 10; j++) {
                const val = matrix[i][j];
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const isDiagonal = (i === j);

                let bgColor;
                if (isDiagonal) {
                    const c = CLASS_COLORS[i];
                    bgColor = c.border.replace('1)', `${0.15 + intensity * 0.55})`);
                } else {
                    bgColor = `rgba(255, 100, 100, ${intensity * 0.4})`;
                }

                const textColor = intensity > 0.4 ? 'white' : 'rgba(255,255,255,0.5)';

                html += `<div class="cm10-cell ${isDiagonal ? 'cm10-diag' : ''}" style="background:${bgColor}; color:${textColor}">${val}</div>`;
            }
        }

        html += '</div>';
        container.innerHTML = html;
    }

    // ================================
    // PER-CLASS ACCURACY BAR CHART
    // ================================
    function renderClassAccuracy(perClassAcc) {
        const ctx = document.getElementById('classAccuracyChart');
        if (!ctx || !perClassAcc) return;
        if (classAccChart) classAccChart.destroy();

        const labels = [];
        const values = [];
        const bgColors = [];
        const borderColors = [];

        for (let i = 0; i < 10; i++) {
            labels.push(`${i}`);
            values.push(perClassAcc[String(i)] || 0);
            bgColors.push(CLASS_COLORS[i].bg);
            borderColors.push(CLASS_COLORS[i].border);
        }

        classAccChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Accuracy %',
                    data: values,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false,
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
                            title: (items) => `Digit ${items[0].label}`,
                            label: (ctx) => `Accuracy: ${ctx.parsed.y.toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Digit Class', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { display: false },
                    },
                    y: {
                        title: { display: true, text: 'Accuracy %', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        min: 80,
                        max: 100,
                        ticks: { font: { size: 9 } }
                    }
                }
            }
        });
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
                        title: { display: true, text: 'Weight Value', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { display: false },
                        ticks: { maxTicksLimit: 15, font: { size: 8 }, maxRotation: 0 }
                    },
                    y: {
                        title: { display: true, text: 'Frequency', font: { family: "'Space Grotesk'", weight: 'bold', size: 10 }, color: '#988ca0' },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { font: { size: 9 } }
                    }
                }
            }
        });

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
            tbody.innerHTML = '<tr><td colspan="4" class="empty-history">No predictions yet. Upload an image to start.</td></tr>';
            return;
        }

        tbody.innerHTML = history.map(h => `
            <tr>
                <td class="td-id">${h.id}</td>
                <td class="td-pred">
                    <span class="digit-badge" style="background:${CLASS_COLORS[h.predicted].bg}; border:1px solid ${CLASS_COLORS[h.predicted].border}">${h.predicted}</span>
                </td>
                <td class="td-conf">${h.confidence}%</td>
                <td class="text-right td-time">${h.time_ms}ms</td>
            </tr>
        `).join('');
    }
});
