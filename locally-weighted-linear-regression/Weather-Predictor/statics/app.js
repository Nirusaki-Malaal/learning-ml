// Delhi Weather Predictor - LWLR Features
const features = [
    { id: 'month', name: 'Month', desc: 'Month of the year (1-12).', detail: 'Seasons heavily influence Delhi temperature.' },
    { id: 'day', name: 'Day', desc: 'Day of the month (1-31).', detail: 'Combined with month to determine seasonal position.' },
    { id: 'humidity', name: 'Humidity', desc: 'Relative humidity percentage.', detail: 'Higher humidity often means monsoon season.' },
    { id: 'windspeed', name: 'Wind Speed', desc: 'Wind speed in km/h.', detail: 'Wind can moderate temperatures.' },
    { id: 'sealevelpressure', name: 'Pressure', desc: 'Sea level atmospheric pressure (hPa).', detail: 'Pressure systems affect weather patterns.' }
];

let currentActualTemp = null;
let optimalBandwidth = 0.5;

// Calculate day of year from month and day
function calculateDayOfYear(month, day) {
    const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let dayOfYear = 0;
    for (let i = 0; i < month - 1; i++) {
        dayOfYear += daysInMonth[i];
    }
    dayOfYear += day;
    return dayOfYear;
}

// Convert day of year back to month and day
function dayOfYearToMonthDay(dayOfYear) {
    const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let month = 1;
    let remainingDays = dayOfYear;
    
    for (let i = 0; i < daysInMonth.length; i++) {
        if (remainingDays <= daysInMonth[i]) {
            return { month: i + 1, day: remainingDays };
        }
        remainingDays -= daysInMonth[i];
        month++;
    }
    return { month: 12, day: 31 };
}

document.addEventListener('DOMContentLoaded', async () => {
    const randomBtn = document.getElementById('randomBtn');
    const predictBtn = document.getElementById('predictBtn');
    const valueDisplay = document.getElementById('predictedValue');
    const explainMsg = document.getElementById('explainMsg');
    const bandwidthSlider = document.getElementById('bandwidth');
    const bandwidthValue = document.getElementById('bandwidthValue');
    const actualTempCard = document.getElementById('actualTempCard');
    const actualTempDisplay = document.getElementById('actualTemp');
    const errorDisplay = document.getElementById('predictionError');

    // Fetch optimal bandwidth from server
    try {
        const response = await fetch('/api/optimal-bandwidth');
        const data = await response.json();
        if (data.optimal_bandwidth) {
            optimalBandwidth = data.optimal_bandwidth;
            if (bandwidthSlider) {
                bandwidthSlider.value = optimalBandwidth;
                if (bandwidthValue) {
                    bandwidthValue.textContent = optimalBandwidth.toFixed(2);
                }
            }
        }
    } catch (err) {
        console.log('Using default bandwidth');
    }

    // Bandwidth slider update
    if (bandwidthSlider && bandwidthValue) {
        bandwidthSlider.addEventListener('input', () => {
            bandwidthValue.textContent = parseFloat(bandwidthSlider.value).toFixed(2);
        });
    }

    // Feature explainers
    features.forEach(f => {
        const input = document.getElementById(f.id);
        if (input && explainMsg) {
            const showExplainer = () => {
                explainMsg.innerHTML = `<strong style="color:#ffcc4d">${f.name}</strong>: ${f.desc} <em>${f.detail}</em>`;
            };
            input.addEventListener('focus', showExplainer);
        }
    });

    // Info button click handlers (for better visibility on mobile)
    features.forEach(f => {
        const input = document.getElementById(f.id);
        if (input) {
            const label = input.closest('.space-y-2');
            if (label) {
                const infoIcon = label.querySelector('.material-symbols-outlined.cursor-pointer');
                if (infoIcon && explainMsg) {
                    infoIcon.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        explainMsg.innerHTML = `<strong style="color:#ffcc4d">${f.name}</strong>: ${f.desc} <em>${f.detail}</em>`;
                        explainMsg.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    });
                }
            }
        }
    });

    // Load random sample
    if (randomBtn) {
        randomBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            const icon = randomBtn.querySelector('.material-symbols-outlined');
            if (icon) icon.classList.add('animate-spin');
            try {
                const response = await fetch('/random');
                const data = await response.json();
                
                // Fill form fields
                if (data.month !== undefined) {
                    document.getElementById('month').value = data.month;
                }
                if (data.dayofyear !== undefined) {
                    // Convert dayofyear to day of month
                    const { month, day } = dayOfYearToMonthDay(data.dayofyear);
                    const dayInput = document.getElementById('day');
                    if (dayInput) {
                        dayInput.value = day;
                    }
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
                    if (actualTempDisplay) {
                        actualTempDisplay.textContent = data.actual_temp.toFixed(1);
                    }
                    if (actualTempCard) {
                        actualTempCard.classList.remove('hidden');
                    }
                }
                
                // Reset prediction display
                if (valueDisplay) {
                    valueDisplay.textContent = '--';
                }
                if (errorDisplay) {
                    errorDisplay.textContent = '--';
                }
                
            } catch (err) {
                console.error('Failed to load random sample:', err);
            } finally {
                const icon = randomBtn.querySelector('.material-symbols-outlined');
                if (icon) icon.classList.remove('animate-spin');
            }
        });
    }

    // Predict button
    if (predictBtn) {
        predictBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            
            const month = parseInt(document.getElementById('month').value) || 5;
            const day = parseInt(document.getElementById('day').value) || 15;
            const dayofyear = calculateDayOfYear(month, day);
            
            const payload = {
                month: month,
                dayofyear: dayofyear,
                humidity: parseFloat(document.getElementById('humidity').value) || 45,
                windspeed: parseFloat(document.getElementById('windspeed').value) || 15,
                sealevelpressure: parseFloat(document.getElementById('sealevelpressure').value) || 1010,
                bandwidth: parseFloat(bandwidthSlider?.value) || optimalBandwidth
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
                if (valueDisplay) {
                    const startVal = parseFloat(valueDisplay.textContent) || 0;
                    animateValue(valueDisplay, startVal, data.prediction, 800);
                }
                
                // Show error if we have actual temp
                if (currentActualTemp !== null && errorDisplay) {
                    const error = Math.abs(data.prediction - currentActualTemp);
                    setTimeout(() => {
                        errorDisplay.textContent = error.toFixed(1);
                    }, 800);
                }
                
            } catch (err) {
                console.error('Prediction failed:', err);
                if (valueDisplay) {
                    valueDisplay.textContent = 'Error';
                }
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
