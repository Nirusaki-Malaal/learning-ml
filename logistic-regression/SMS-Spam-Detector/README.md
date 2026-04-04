# 📱 SMS Spam Detector

A modern, custom-built SMS spam detection system using **Logistic Regression** optimized with **Newton's Method** — built entirely from scratch using NumPy, no scikit-learn!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎯 **Custom Logistic Regression** - Implemented from scratch using NumPy
- ⚡ **Newton's Method Optimization** - Second-order optimization with Hessian matrix inversion
- 🔤 **Custom TF-IDF Vectorizer** - Hand-coded feature extraction (no sklearn)
- 🎨 **Modern Dark UI** - Glassmorphism design with smooth animations
- 📊 **Interactive Analytics Dashboard** - Deep dive into model performance
- 📈 **Real-time Predictions** - Instant spam/ham classification with probability scores
- 🎲 **Sample SMS Loader** - Pre-loaded spam and ham examples

## 🚀 Demo Features

### Main Detector
- Real-time SMS classification
- Probability bars with smooth animations
- Load sample messages (spam/ham)
- TF-IDF feature extraction
- Gradient-based result visualization

### Analytics Dashboard
- **Animated Statistics** - Total messages, spam/ham counts, accuracy
- **Class Distribution Chart** - Interactive donut chart with ratio display
- **Sigmoid Visualization** - Decision boundary and probability curve
- **Word Clouds** - Size-weighted spam/ham indicator words
- **Feature Importance** - Toggle between spam and ham word rankings
- **Message Length Analysis** - Comparative length distribution
- **Confusion Matrix** - TP, TN, FP, FN with precision, recall, F1-score
- **Model Architecture** - 3-stage pipeline breakdown

## 🛠️ Tech Stack

**Backend:**
- FastAPI (REST API)
- NumPy (Custom ML implementation)
- Pandas (Data handling)

**Frontend:**
- HTML5 / TailwindCSS
- Chart.js (Data visualization)
- Vanilla JavaScript

**ML Pipeline:**
1. **TF-IDF Vectorizer** - Custom term frequency-inverse document frequency implementation
2. **Logistic Regression** - Binary classification with sigmoid activation
3. **Newton's Method** - Fast convergence with L2 regularization

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Email-Spam-Detector
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify dataset**
Ensure `spam.csv` is in the root directory with columns:
- `label` - "spam" or "ham"
- `text` - SMS message content

## 🎯 Usage

### Start the server
```bash
uvicorn main:app --reload
```

The application will be available at:
- **Detector**: http://localhost:8000
- **Analytics**: http://localhost:8000/analytics

### Using the Detector

1. Open http://localhost:8000
2. Paste an SMS message or click "Load Sample"
3. Click "Check for Spam"
4. View the classification result with probability scores

### Exploring Analytics

1. Navigate to http://localhost:8000/analytics
2. View model statistics and performance metrics
3. Explore word importance and feature distributions
4. Analyze the confusion matrix and decision boundary

## 📊 Model Performance

- **Algorithm**: Custom Logistic Regression
- **Optimization**: Newton's Method (2nd order)
- **Features**: 1,500 TF-IDF features
- **Training Epochs**: 15 (Newton's method converges fast!)
- **Test Accuracy**: ~97%+ (varies based on train/test split)

## 🧠 How It Works

### TF-IDF Vectorization
Converts SMS text into numerical features:
- **TF** (Term Frequency): How often a word appears in a message
- **IDF** (Inverse Document Frequency): How unique a word is across all messages
- Result: Sparse feature vectors emphasizing important words

### Logistic Regression
Binary classifier using sigmoid function:
```
P(spam) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```
- If P(spam) ≥ 0.5 → Classify as Spam
- If P(spam) < 0.5 → Classify as Ham

### Newton's Method
Fast optimization using the Hessian matrix:
```
θ_{new} = θ_{old} + H⁻¹∇L
```
- Converges in ~15 epochs (vs 1000+ for gradient descent)
- Uses L2 regularization for numerical stability
- Dramatically faster than first-order methods

## 📁 Project Structure

```
Email-Spam-Detector/
├── main.py                 # FastAPI backend
├── app.py                  # Alternative entry point
├── spam.csv               # Dataset
├── requirements.txt       # Dependencies
├── plugins/
│   ├── Logistic_Regression.py  # Custom LR implementation
│   └── TF_IDFVectorizer.py     # Custom TF-IDF implementation
├── templates/
│   ├── index.html         # Main detector UI
│   └── analytics.html     # Analytics dashboard
└── statics/
    ├── style.css          # Global styles
    └── app.js             # Frontend logic
```

## 🎨 UI Highlights

- **Dark Theme** - Easy on the eyes with glassmorphism effects
- **Material Icons** - Modern icon system
- **Responsive Design** - Mobile-friendly layout
- **Smooth Animations** - Counting animations, progress bars, transitions
- **Interactive Charts** - Hover effects and data exploration

## 📝 Dataset

Uses the SMS Spam Collection dataset:
- **Total Messages**: ~5,572
- **Spam**: ~747 (13.4%)
- **Ham**: ~4,825 (86.6%)

Format: CSV with `label` and `text` columns

## 🔧 Training from Scratch

To retrain the model:

```python
python app.py
```

This will:
1. Load `spam.csv`
2. Vectorize messages with TF-IDF
3. Train using Newton's method (15 epochs)
4. Print accuracy on test set
5. Save trained model in memory

## 🚦 API Endpoints

- `GET /` - Main detector page
- `GET /analytics` - Analytics dashboard
- `POST /predict` - Predict spam/ham
  - Body: `sms_text` (form data)
  - Returns: `prediction`, `spam_probability`, `ham_probability`
- `GET /api/analytics` - Get analytics data (JSON)

## 🤝 Contributing

This is a learning project! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - feel free to use this project for learning!

## 🎓 Educational Value

Perfect for learning:
- ✅ Custom ML implementation (no black boxes!)
- ✅ Newton's Method optimization
- ✅ FastAPI backend development
- ✅ Modern frontend with TailwindCSS
- ✅ Data visualization with Chart.js
- ✅ Model evaluation metrics

## 🙏 Acknowledgments

- Dataset: SMS Spam Collection
- Icons: Google Material Symbols
- Charts: Chart.js
- CSS Framework: TailwindCSS

## 📧 Contact

Built with 💙 using custom NumPy implementations - no sklearn shortcuts!

---

**Note**: This is a learning project demonstrating ML fundamentals from scratch. For production use, consider battle-tested libraries like scikit-learn.
