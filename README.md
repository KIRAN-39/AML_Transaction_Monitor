# 🔍 AML Transaction Monitoring System

An intelligent Anti-Money Laundering detection system using rule-based algorithms and machine learning to identify suspicious financial transactions.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **4 Detection Methods**
  - 5 Rule-based algorithms (High-Value, Velocity, Structuring, Dormant Account, Round Numbers)
  - Isolation Forest (Unsupervised ML)
  - K-Means Clustering
  - Hybrid approach (Rules + ML)

- **Interactive Dashboard**
  - Real-time transaction monitoring
  - Risk-based alert filtering
  - Interactive analytics with Plotly charts
  - Streamlit web interface

- **Professional Reporting**
  - Excel reports with multiple sheets
  - CSV exports for external systems
  - Compliance-ready text reports
  - Visualization charts

- **High Performance**
  - Processes 50,000+ transactions
  - 86% F1-Score (Hybrid method)
  - Real-time risk scoring

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/AML-Transaction-Monitor.git
cd AML-Transaction-Monitor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Generate sample data
```bash
python src/data_generator.py
```

4. Run the dashboard
```bash
streamlit run src/dashboard.py
```

Open browser at `http://localhost:8501`

## 📖 Usage

### Generate Sample Data (50,000 transactions)
```bash
python src/data_generator.py
```

### Run Complete Detection Analysis
```bash
python src/test_detection_system.py
```

### Compare All Detection Methods
```bash
python src/evaluate_models.py
```

### Launch Interactive Dashboard
```bash
streamlit run src/dashboard.py
```

## 🏗️ System Architecture
```
Transaction Data Input
         ↓
   Detection Layer
    ├── Rules (5 patterns)
    └── ML (Isolation Forest, K-Means)
         ↓
    Risk Scoring
         ↓
   Alert Management
         ↓
 Dashboard & Reports
```

## 📊 Performance Results

| Method | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Rules  | 78%       | 82%    | 80%      |
|Isolation           |        |          |
| Forest | 75%       | 85%    | 80%      |
| K-Means| 72%       | 88%    | 79%      |
| Hybrid | 81%       | 92%    | 86%      |
|________|___________|________|__________|
## 🔍 Detection Methods

### Rule-Based
1. **High-Value** - Transactions > $10,000
2. **Velocity** - Rapid transaction patterns (10+ in 24h)
3. **Structuring** - Transaction splitting near $10k threshold
4. **Dormant Account** - Old accounts suddenly active
5. **Round Numbers** - Suspicious exact amounts

### Machine Learning
- **Isolation Forest** - Anomaly detection
- **K-Means** - Clustering-based outlier detection

## 🛠️ Tech Stack

- Python 3.9+
- Pandas & NumPy - Data processing
- Scikit-learn - Machine learning
- Streamlit - Web dashboard
- Plotly - Interactive visualizations
- SQLite - Alert storage

## 📂 Project Structure
```
AML-Transaction-Monitor/
├── src/              # Source code (10 modules)
├── data/             # Generated transaction data
├── models/           # Trained ML models
├── reports/          # Generated reports
└── screenshots/      # Dashboard previews
```

## 📸 Screenshots

![Model Comparison](screenshots/model_comparison.png)
![Dashboard](screenshots/Dashboard.png)
![Analysis](screenshots/Analysis.png)
![Navigation_Bar](screenshots/Navigation_Bar.png)
![Transaction_monitor](screenshots/Transaction_monitor.png)

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📄 License

MIT License - see LICENSE file

## 👨‍💻 Author

**[Your Name]**
- GitHub: [@KIRAN-39](https://github.com/KIRAN-39)
- Email: kbn2024is@example.com


## ⚠️ Disclaimer

This is a demonstration project. For production use in financial institutions, ensure compliance with local AML regulations (FinCEN, FATF, etc.).

---

⭐ Star this repo if you found it helpful!