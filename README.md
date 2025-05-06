# AutoProcess

**AutoProcess** is a modular Python package for automated data preprocessing, designed to streamline data preparation tasks such as cleaning, transformation, feature engineering, and more.

---

## 📁 Project Structure

```
AutoProcess/
├── generate/
│   └── src/
│       └── autoprocess/
│           ├── cleaning.py
│           ├── feature_eng.py
│           ├── helper.py
│           ├── transforming.py
│           └── unskew.py
├── LICENSE
├── README.md
└── setup.py
```

---

## ✨ Features Overview

### `cleaning.py` – Data Cleaning

Handles essential data cleaning tasks:
- `remove_nulls()`: Drops or fills missing values based on strategy.
- `remove_outliers()`: Identifies and removes outliers using z-score or IQR.
- `standardize_column_names()`: Ensures column names are consistent and lowercase.
- `drop_duplicates()`: Removes duplicate rows.

### `feature_eng.py` – Feature Engineering

Includes functions to generate new features:
- `create_interaction_terms()`: Generates interaction terms between numerical features.
- `extract_datetime_features()`: Extracts year, month, day, etc., from datetime columns.
- `binning()`: Converts continuous features into categorical bins.

### `helper.py` – Utility Functions

Support functions used throughout the package:
- Logging utilities.
- Data validation checks.
- Reusable decorators or configuration management.

### `transforming.py` – Data Transformation

Prepares data for modeling:
- `encode_categoricals()`: One-hot or label encodes categorical variables.
- `scale_features()`: Scales features using MinMaxScaler, StandardScaler, etc.
- `normalize_data()`: Applies normalization for numerical stability.

### `unskew.py` – Skewness Correction

Deals with highly skewed distributions:
- `log_transform()`, `boxcox_transform()`: Applies skew-reducing transformations.
- Automatically detects and corrects skewness above a defined threshold.

---

## ⚙️ Installation

```bash
pip install autoprocess_iitg
```

---

## 🧪 Usage Example

```python
from autoprocess import cleaning, transforming, feature_eng

df = cleaning.remove_nulls(df)
df = transforming.encode_categoricals(df)
df = feature_eng.extract_datetime_features(df, column='date')
```

---

## 📄 License

This project is licensed under the terms of the included LICENSE file.
