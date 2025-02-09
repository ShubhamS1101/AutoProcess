import google.generativeai as genai
import pandas as pd
import torch
import numpy as np
import os


def gen_des(df: pd.DataFrame, sample_size=2) -> dict:
   
    description = {
        "columns": {},
        "correlation": df.corr(numeric_only=True).round(2).to_dict(),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "sample_rows": df.sample(min(sample_size, len(df)), random_state=42).to_dict(orient='records')
    }

    for col in df.columns:
        col_data = df[col]
        col_info = {
            "dtype": str(col_data.dtype),
            "missing_pct": round(col_data.isna().mean() * 100, 1),
            "unique_count": col_data.nunique(),
            "example_values": col_data.dropna().sample(
                min(5, len(col_data.dropna())), 
                random_state=42
            ).tolist()
        }

        # Numeric column features
        if pd.api.types.is_numeric_dtype(col_data):
            col_info.update({
                "min": round(float(col_data.min()), 2) if not col_data.isna().all() else None,
                "max": round(float(col_data.max()), 2) if not col_data.isna().all() else None,
                "mean": round(float(col_data.mean()), 2) if not col_data.isna().all() else None,
                "std": round(float(col_data.std()), 2) if not col_data.isna().all() else None,
                "skew": round(float(col_data.skew()), 2) if len(col_data.dropna()) > 1 else None
            })

        # Categorical features
        if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            freq = col_data.value_counts(normalize=True).head(3)
            col_info["value_distribution"] = {
                "top_values": freq.index.tolist(),
                "percentages": [round(p*100, 1) for p in freq.values]
            }

        description["columns"][col] = col_info

    return description


def initialize_gemini(api_key: str):
 
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Test the model with a simple prompt to ensure the API key is valid.
        test_response = model.generate_content("Test")
        if not test_response.text:
            raise Exception("No response")
        return model
    except Exception as e:
        print("API not valid")
        return None