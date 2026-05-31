# Week 2 — Weekly Progress Report

**Thesis:** Comparative Evaluation of Classical Machine Learning and Transformer-Based Models for SMS Spam Detection
**Author:** Michał Pokracki
**Period:** Week 2
**Date:** 2026-05-31

---

## 1. Progress This Week

The focus of week 2 was moving from planning into the practical part of the
project and producing the first working, reproducible code in the repository.

Components implemented:

- **Repository structure** — organised the project into `notebooks/`, `src/`,
  `results/` and `docs/`, added `requirements.txt` and expanded the `README`
  with setup instructions and a status section.
- **Exploratory Data Analysis (`01_exploratory_analysis.ipynb`)** — loads the
  UCI SMS Spam Collection (5,572 messages), analyses the class distribution,
  message-length characteristics, representative samples, and data-quality
  checks (missing values, duplicates).
- **Baseline classifier (`02_baseline_naive_bayes.ipynb`)** — a Multinomial
  Naive Bayes model with TF-IDF features (unigrams + bigrams, `min_df=2`,
  `max_features=10000`), a stratified 80/20 train/test split, and evaluation on
  all five predictive metrics plus a confusion matrix.

Both notebooks were executed end-to-end; their outputs (plots, metrics,
classification report) are committed to the repository, so the results are
visible without re-running anything.

### Baseline results (Multinomial Naive Bayes)

| Metric | Value |
|---|---|
| Accuracy | 97.04% |
| Precision | 100.00% |
| Recall | 77.85% |
| F1-score | 0.876 |
| ROC-AUC | 0.988 |

Operational reference: training ~0.003 s, inference ~0.0002 ms/message.
The confusion matrix shows 0 false positives (no ham flagged as spam) and 33 of
149 spam messages missed — the high-precision / lower-recall behaviour expected
of Naive Bayes on an imbalanced dataset (~87% ham / ~13% spam). This establishes
the reference point against which SVM, Logistic Regression and DistilBERT will
later be compared.

## 2. Takeaways from the First Demo

The first demo was **rescheduled to week 3** by agreement with the supervisor,
so that it can be presented together with the third weekly report. As a result,
no demo feedback has been recorded this week.

To prepare for it, week 2 deliberately prioritised having genuinely working,
reproducible artefacts (executed notebooks with real metrics) rather than
code that merely exists in the repository. This gives the week-3 demo a concrete
baseline to show and discuss.

## 3. Problems Encountered and How They Were Addressed

- **Dataset could not be loaded via `ucimlrepo`.** The planned approach used
  `fetch_ucirepo(id=228)`, but this raises `DatasetNotFoundError` — the SMS Spam
  Collection exists in the UCI repository but is not exposed for direct import
  (it has no standardised CSV). *Resolution:* the data-loading cells were
  rewritten to download the official UCI archive (ZIP), extract the
  tab-separated `SMSSpamCollection` file, and parse it with pandas. This made
  both notebooks actually executable.
- **SSL verification when downloading from UCI on macOS.** A plain `urllib`
  request failed certificate verification. *Resolution:* the request uses an SSL
  context backed by `certifi`'s certificate bundle; `certifi` was added to
  `requirements.txt`.
- **Local environment not set up.** scikit-learn, pandas and the rest were not
  installed. *Resolution:* created an isolated virtual environment and installed
  everything from `requirements.txt`; verified all imports succeed.
- **No GitHub authentication on the machine.** The initial push failed
  (`could not read Username`). *Resolution:* installed and authenticated the
  GitHub CLI (`gh auth login`), which configured the git credential helper and
  enabled the push.

## 4. Updated Work Plan — Week 3

- Add two further classical baselines: **Linear SVM** and **Logistic
  Regression**, reusing the shared TF-IDF preprocessing.
- Extract the preprocessing/evaluation logic from the notebooks into reusable
  modules under `src/`.
- Begin **fine-tuning DistilBERT** (likely on Google Colab for GPU access).
- Define the **operational metrics protocol** (training time, inference latency,
  model size, memory footprint) under a unified benchmarking environment.
- **Conduct the first demo** and record the supervisor's feedback.

## 5. Artefacts

Repository: <https://github.com/michalpokracki/master-thesis-sms-spam-classification>
Commit for this report: [`bbb4df0`](https://github.com/michalpokracki/master-thesis-sms-spam-classification/commit/bbb4df0bbef0d986fe7112651bdccd1e6ba7de05)

Code:

- EDA notebook: [`notebooks/01_exploratory_analysis.ipynb`](https://github.com/michalpokracki/master-thesis-sms-spam-classification/blob/main/notebooks/01_exploratory_analysis.ipynb)
- Baseline notebook: [`notebooks/02_baseline_naive_bayes.ipynb`](https://github.com/michalpokracki/master-thesis-sms-spam-classification/blob/main/notebooks/02_baseline_naive_bayes.ipynb)
- Dependencies: [`requirements.txt`](https://github.com/michalpokracki/master-thesis-sms-spam-classification/blob/main/requirements.txt)

Screenshots (figures):

- Class distribution: [`results/figures/class_distribution.png`](https://github.com/michalpokracki/master-thesis-sms-spam-classification/blob/main/results/figures/class_distribution.png)
- Message-length distributions: [`results/figures/message_length_distributions.png`](https://github.com/michalpokracki/master-thesis-sms-spam-classification/blob/main/results/figures/message_length_distributions.png)
- Confusion matrix: [`results/figures/confusion_matrix.png`](https://github.com/michalpokracki/master-thesis-sms-spam-classification/blob/main/results/figures/confusion_matrix.png)

Demo video: not applicable this week (demo rescheduled to week 3).
