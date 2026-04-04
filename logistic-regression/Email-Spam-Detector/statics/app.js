// Sample SMSs for demo
const sampleSMSs = {
    spam: [
        "CONGRATULATIONS! You've been selected to receive a FREE iPhone 15! Click here NOW to claim your prize before it expires. Limited time offer!",
        "URGENT: Your account has been compromised. Click this link immediately to verify your identity and secure your funds: http://totally-legit-bank.com",
        "Make $5000 weekly working from home! No experience needed. Reply with your personal information to get started NOW!",
        "Dear Winner, You have won $1,000,000 in our lottery. Send us your bank details to claim your prize money today!",
        "Hot singles in your area want to meet you! Click here for FREE access to exclusive dating profiles. Don't miss out!"
    ],
    ham: [
        "Hi team, just a reminder about tomorrow's meeting at 10 AM. Please review the attached documents beforehand. See you there!",
        "Hey! Are we still on for lunch on Friday? Let me know what time works best for you. Looking forward to catching up!",
        "Thank you for your order #12345. Your package has been shipped and should arrive within 3-5 business days.",
        "Dear Student, Your assignment submission has been received. Grades will be posted by end of week. Best regards, Professor Smith",
        "Just wanted to check in - how's the project going? Let me know if you need any help with the database integration."
    ]
};

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('spamForm');
    const SMSInput = document.getElementById('sms_text');
    const predictBtn = document.getElementById('predictBtn');
    const loadSampleBtn = document.getElementById('loadSampleBtn');
    const predictionResult = document.getElementById('predictionResult');
    const resultIcon = document.getElementById('resultIcon');
    const spamBar = document.getElementById('spamBar');
    const hamBar = document.getElementById('hamBar');
    const spamProbText = document.getElementById('spamProbText');
    const hamProbText = document.getElementById('hamProbText');
    const explainMsg = document.getElementById('explainMsg');

    // Load random sample
    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            loadSampleBtn.querySelector('.material-symbols-outlined').classList.add('animate-spin');
            
            setTimeout(() => {
                // Randomly choose spam or ham
                const type = Math.random() > 0.5 ? 'spam' : 'ham';
                const samples = sampleSMSs[type];
                const randomSMS = samples[Math.floor(Math.random() * samples.length)];
                
                SMSInput.value = randomSMS;
                explainMsg.innerHTML = `<span class="text-primary">Sample loaded!</span> This is a <strong class="${type === 'spam' ? 'text-red-400' : 'text-green-400'}">${type}</strong> SMS example.`;
                
                loadSampleBtn.querySelector('.material-symbols-outlined').classList.remove('animate-spin');
            }, 300);
        });
    }

    // Clear explainer on input
    if (SMSInput) {
        SMSInput.addEventListener('focus', () => {
            explainMsg.innerHTML = '<span class="text-primary">TF-IDF Vectorization</span>: Your SMS will be converted to numerical features based on term frequency and importance.';
        });
    }

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const SMSText = SMSInput.value.trim();
        if (!SMSText) {
            explainMsg.innerHTML = '<span class="text-red-400">Please enter some message text to analyze.</span>';
            return;
        }

        // Disable button and show loading
        predictBtn.disabled = true;
        const originalHTML = predictBtn.innerHTML;
        predictBtn.innerHTML = '<span class="material-symbols-outlined animate-spin text-2xl">refresh</span> Analyzing...';

        try {
            const formData = new FormData();
            formData.append('sms_text', SMSText);
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Update result display
            const isSpam = data.prediction === "Spam";
            
            if (isSpam) {
                predictionResult.innerHTML = `<span class="spam-gradient">🚫 SPAM</span>`;
                resultIcon.textContent = 'warning';
                resultIcon.className = 'material-symbols-outlined text-6xl text-red-400';
            } else {
                predictionResult.innerHTML = `<span class="ham-gradient">✅ SAFE</span>`;
                resultIcon.textContent = 'verified';
                resultIcon.className = 'material-symbols-outlined text-6xl text-green-400';
            }
            
            // Animate probability bars
            requestAnimationFrame(() => {
                spamBar.style.width = `${data.spam_probability}%`;
                hamBar.style.width = `${data.ham_probability}%`;
                
                // Animate text counters
                animateValue(spamProbText, 0, data.spam_probability, 700, '%');
                animateValue(hamProbText, 0, data.ham_probability, 700, '%');
            });
            
            explainMsg.innerHTML = '';
            
        } catch (err) {
            console.error('Prediction failed:', err);
            predictionResult.innerHTML = '<span class="text-red-400">Error</span>';
            explainMsg.innerHTML = '<span class="text-red-400">Failed to analyze message. Is the server running?</span>';
        } finally {
            predictBtn.innerHTML = originalHTML;
            predictBtn.disabled = false;
        }
    });

    function animateValue(element, start, end, duration, suffix = '') {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const easeOutProgress = 1 - Math.pow(1 - progress, 3);
            const current = start + easeOutProgress * (end - start);
            element.textContent = current.toFixed(1) + suffix;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});