# Master's Thesis — SMS Spam Classification

This repository contains the code, experiments, and documentation for my Master's thesis in Computer Science.

## Topic

Comparative Evaluation of Classical Machine Learning and Transformer-Based Models for SMS Spam Detection.

## About

The project explores how classical machine learning models compare with modern transformer-based approaches in the task of SMS spam classification, with a focus on the trade-off between predictive performance and deployment cost.

> **Terminology.** Each SMS message is labelled either **spam** (an unsolicited message) or **ham**. *Ham* is the standard term in the spam-filtering literature for a legitimate, non-spam message — an ordinary message the user actually wants to receive.

## Tech Stack

Python, scikit-learn, Hugging Face Transformers, PyTorch, NLTK, pandas, NumPy, Matplotlib, Streamlit.

## Repository Structure

```
.
├── notebooks/     # exploratory and experimental notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_baseline_naive_bayes.ipynb
│   └── 03_classical_models_comparison.ipynb
├── src/           # reusable modules (data loading, preprocessing, models, evaluation)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── train_classical.py
├── tests/         # automated test suite (pytest)
│   └── test_pipeline.py
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
```

## Running the comparison

```bash
# Train and compare all classical models (prints a results table)
python -m src.train_classical

# Run the notebooks
jupyter notebook
```

## Running the tests

```bash
pytest -v
```

## Status

Work in progress (week 3 of 8).

Completed so far:

- Topic approved and proposal finalised.
- Exploratory data analysis of the UCI SMS Spam Collection.
- Three classical baselines (Multinomial Naive Bayes, Linear SVM, Logistic Regression with TF-IDF) trained, evaluated, and compared through shared, tested modules.
- Automated test suite covering data integrity, the split, the vectorizer, model training, and reproducibility.

Next:

- Fine-tune DistilBERT and add it to the comparison.
- Add model size and memory footprint to the operational metrics.

## Author

Michał Pokracki
