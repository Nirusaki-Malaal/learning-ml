// Delhi Weather Predictor - LWLR Features
const features = [
    { id: 'month', name: 'Month', desc: 'Month of the year (1-12).', detail: 'Seasons heavily influence Delhi temperature.' },
    { id: 'dayofyear', name: 'Day of Year', desc: 'Day number in the year (1-365).', detail: 'Captures seasonal temperature patterns.' },
    { id: 'humidity', name: 'Humidity', desc: 'Relative humidity percentage.', detail: 'Higher humidity often means monsoon season.' },
    { id: 'windspeed', name: 'Wind Speed', desc: 'Wind speed in km/h.', detail: 'Wind can moderate temperatures.' },
    { id: 'sealevelpressure', name: 'Pressure', desc: 'Sea level atmospheric pressure (hPa).', detail: 'Pressure systems affect weather patterns.' }
];

let currentActualTemp = null;

document.addEventListener('DOMContentLoaded', () => {
    const randomBtn = document.getElementById('randomBtn');
    const predictBtn = document.getElementById('predictBtn');
    const valueDisplay = document.getElementById('predictedValue');
    const explainMsg = document.getElementById('explainMsg');
    const bandwidthSlider = document.getElementById('bandwidth');
    const bandwidthValue = document.getElementById('bandwidthValue');
    const actualTempCard = document.getElementById('actualTempCard');
    const actualTempDisplay = document.getElementById('actualTemp');
    const errorDisplay = document.getElementById('predictionError');

    // Bandwidth slider update
    if (bandwidthSlider && bandwidthValue) {
        bandwidthSlider.addEventListener('input', () => {
            bandwidthValue.textContent = parseFloat(bandwidthSlider.value).toFixed(2);
        });
    }

    // Feature explainers
    features.forEach(f => {
        const input = document.getElementById(f.id);
        if (input) {
            const showExplainer = () => {
                explainMsg.innerHTML = `<strong style="color:#ffcc4d">${f.name}</strong>: ${f.desc} <em>${f.detail}</em>`;
            };
            input.addEventListener('focus', showExplainer);
        }
    });

    // Load random sample
    if (randomBtn) {
        randomBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            randomBtn.querySelector('.material-symbols-outlined').classList.add('animate-spin');
            try {
                const response = await fetch('/random');
                const data = await response.json();
                
                // Fill form fields
                if (data.month !== undefined) {
                    document.getElementById('month').value = data.month;
                }
                if (data.dayofyear !== undefined) {
                    document.getElementById('dayofyear').value = data.dayofyear;
                }
                if (data.humidity !== undefined) {
                    document.getElementById('humidity').value = Math.round(data.humidity);
                }
                if (data.windspeed !== undefined) {
                    document.getElementById('windspeed').value = Math.round(data.windspeed * 10) / 10;
                }
                if (data.sealevelpressure !== undefined) {
                    document.getElementById('sealevelpressure').value = Math.round(data.sealevelpressure);
                }
                
                // Store and show actual temperature
                if (data.actual_temp !== undefined) {
                    currentActualTemp = data.actual_temp;
                    actualTempDisplay.textContent = data.actual_temp.toFixed(1);
                    actualTempCard.classList.remove('hidden');
                }
                
                // Reset prediction display
                valueDisplay.textContent = '--';
                errorDisplay.textContent = '--';
                
            } catch (err) {
                console.error('Failed to load random sample:', err);
            } finally {
                randomBtn.querySelector('.material-symbols-outlined').classList.remove('animate-spin');
            }
        });
    }

    // Predict button
    if (predictBtn) {
        predictBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            
            const payload = {
                month: parseInt(document.getElementById('month').value) || 5,
                dayofyear: parseInt(document.getElementById('dayofyear').value) || 150,
                humidity: parseFloat(document.getElementById('humidity').value) || 45,
                windspeed: parseFloat(document.getElementById('windspeed').value) || 15,
                sealevelpressure: parseFloat(document.getElementById('sealevelpressure').value) || 1010,
                bandwidth: parseFloat(bandwidthSlider?.value) || 0.5
            };

            const originalHTML = predictBtn.innerHTML;
            predictBtn.innerHTML = '<span class="material-symbols-outlined animate-spin">refresh</span> Predicting...';
            predictBtn.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                
                // Animate the value
                const startVal = parseFloat(valueDisplay.textContent) || 0;
                animateValue(valueDisplay, startVal, data.prediction, 800);
                
                // Show error if we have actual temp
                if (currentActualTemp !== null) {
                    const error = Math.abs(data.prediction - currentActualTemp);
                    setTimeout(() => {
                        errorDisplay.textContent = error.toFixed(1);
                    }, 800);
                }
                
            } catch (err) {
                console.error('Prediction failed:', err);
                valueDisplay.textContent = 'Error';
            } finally {
                predictBtn.innerHTML = originalHTML;
                predictBtn.disabled = false;
            }
        });
    }

    function animateValue(obj, start, end, duration) {
        if (isNaN(start)) start = 0;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const easeOutProgress = 1 - Math.pow(1 - progress, 3);
            const current = start + easeOutProgress * (end - start);
            obj.innerHTML = current.toFixed(1);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
