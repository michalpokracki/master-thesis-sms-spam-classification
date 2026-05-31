# Master's Thesis — SMS Spam Classification

This repository contains the code, experiments, and documentation for my Master's thesis in Computer Science.

## Topic

Comparative Evaluation of Classical Machine Learning and Transformer-Based Models for SMS Spam Detection.

## About

The project explores how classical machine learning models compare with modern transformer-based approaches in the task of SMS spam classification, with a focus on the trade-off between predictive performance and deployment cost.

## Tech Stack

Python, scikit-learn, Hugging Face Transformers, PyTorch, NLTK, pandas, NumPy, Matplotlib, Streamlit.

## Repository Structure

```
.
├── notebooks/     # exploratory and experimental notebooks
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_baseline_naive_bayes.ipynb
├── src/           # source code (preprocessing, models, evaluation)
├── results/       # metrics, plots, trained models
├── docs/          # documentation and weekly reports
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/michalpokracki/master-thesis-sms-spam-classification.git
cd master-thesis-sms-spam-classification

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook
```

## Status

Work in progress (week 2 of 8).

Completed so far:
- Topic approved and proposal finalised.
- Exploratory data analysis of the UCI SMS Spam Collection.
- First baseline classifier (Multinomial Naive Bayes with TF-IDF) trained and evaluated.

## Author

Michał Pokracki
