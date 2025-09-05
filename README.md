# ML Calorie Predictor 🥗

A reproducible machine learning pipeline using DVC, Git, and Python to predict food calories based on nutritional features.

## 🔧 Features

- Modular pipeline with `preprocess`, `train`, and `evaluate` stages
- Configurable via `params.yaml`
- Experiment tracking with `dvc exp`
- Metrics visualization with `dvc plots`
- Local remote storage for data versioning

## 📁 Project Structure

data/
├── raw/row.csv
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
model/
├── model.pkl
src/
├── preprocess.py
├── train.py
├── evaluate.py
params.yaml
metrics.json
dvc.yaml
requirements.txt
README.md

`<pre><div>`---


## ⚙️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/ml-calorie-predictor.git
cd ml-calorie-predictor

# Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1  # For PowerShell

# Install dependencies
pip install -r requirements.txt</div></pre>
```

## 🧪 Run the Pipeline

dvc repro

dvc exp run --set-param train.test_size=0.3

dvc exp show --sort-by mse --sort-order ascVisualize metrics

dvc plots diff --open

## 📊 Sample Metrics

{
  "r2_score": 0.9006,
  "mse": 0.133
}

## 📦 Requirements

* Python 3.8+
* pandas
* scikit-learn
* pyyaml
* joblib
* DVC
* Git
