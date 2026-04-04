const features = [
    { id: 'crim', name: 'CRIM', desc: 'Per capita crime rate by town.', detail: 'Lower values indicate safer neighborhoods.' },
    { id: 'zn', name: 'ZN', desc: 'Proportion of residential land zoned for lots over 25,000 sq.ft.', detail: 'Higher values mean larger residential plots.' },
    { id: 'indus', name: 'INDUS', desc: 'Proportion of non-retail business acres per town.', detail: 'High industrial areas might lower residential appeal.' },
    { id: 'chas', name: 'CHAS', desc: 'Charles River dummy variable.', detail: 'Proximity to the river typically adds value.' },
    { id: 'nox', name: 'NOX', desc: 'Nitric oxides concentration (parts per 10 million).', detail: 'Higher pollution correlates with lower values.' },
    { id: 'rm', name: 'RM', desc: 'Average number of rooms per dwelling.', detail: 'Strongest predictor - more rooms = higher value.' },
    { id: 'age', name: 'AGE', desc: 'Proportion of owner-occupied units built prior to 1940.', detail: 'Higher percentage may mean older infrastructure.' },
    { id: 'dis', name: 'DIS', desc: 'Weighted distances to five Boston employment centres.', detail: 'Proximity to work hubs increases demand.' },
    { id: 'rad', name: 'RAD', desc: 'Index of accessibility to radial highways.', detail: 'Ease of commute metric.' },
    { id: 'tax', name: 'TAX', desc: 'Full-value property-tax rate per $10,000.', detail: 'High taxes may indicate better services.' },
    { id: 'ptratio', name: 'PTRATIO', desc: 'Pupil-teacher ratio by town.', detail: 'Lower ratio means smaller classes.' },
    { id: 'b', name: 'B', desc: '1000(Bk - 0.63)^2 demographic index.', detail: 'Historical demographic indicator.' },
    { id: 'lstat', name: 'LSTAT', desc: '% lower status of the population.', detail: 'Inversely correlated with house prices.' }
];

document.addEventListener('DOMContentLoaded', () => {
    const randomBtn = document.getElementById('randomBtn');
    const predictBtn = document.getElementById('predictBtn');
    const valueDisplay = document.getElementById('predictedValue');
    const explainMsg = document.getElementById('explainMsg');

    features.forEach(f => {
        const input = document.getElementById(f.id);
        if(input) {
            const showExplainer = () => {
                 explainMsg.innerHTML = `<strong style="color:#00e5ff">${f.name}</strong>: ${f.desc} <em>${f.detail}</em>`;
            };
            input.addEventListener('focus', showExplainer);
            
            // Add click handler for info icons to work on mobile/touch devices
            const label = input.parentElement.querySelector('label .material-symbols-outlined');
            if(label) {
                label.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    showExplainer();
                    input.focus();
                });
                // Make it more obvious it's clickable
                label.style.cursor = 'pointer';
            }
        }
    });

    if(randomBtn) {
        randomBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            try {
                const response = await fetch(`/random`);
                const data = await response.json();
                features.forEach(f => {
                    if(data[f.id] !== undefined) {
                        const inputTarget = document.getElementById(f.id);
                        if(inputTarget) inputTarget.value = data[f.id];
                    }
                });
            } catch (err) {
                console.error(err);
            }
        });
    }

    if(predictBtn) {
        predictBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            const payload = {};
            features.forEach(f => {
                const input = document.getElementById(f.id);
                if(input) payload[f.id] = parseFloat(input.value) || 0;
            });

            const originalText = predictBtn.innerText;
            predictBtn.innerText = 'Predicting...';

            try {
                const response = await fetch(`/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                animateValue(valueDisplay, parseFloat(valueDisplay.innerText), data.prediction, 1000);
            } catch (err) {
                console.error(err);
            } finally {
                predictBtn.innerText = originalText;
            }
        });
    }

    function animateValue(obj, start, end, duration) {
        if(isNaN(start)) start = 0;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const easeOutProgress = 1 - Math.pow(1 - progress, 3);
            const current = start + easeOutProgress * (end - start);
            obj.innerHTML = current.toFixed(2);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
