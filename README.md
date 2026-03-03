# mobile-price-prediction
This project implements a machine learning model to predict the price of mobile phones based on their specifications. The model preprocesses raw mobile data, performs feature engineering, and predicts prices using a stacking regressor pipeline.

## Overview

The goal of this project is to accurately predict the price of mobile phones using specifications such as company, OS, chipset, camera, display, and performance metrics.
Key highlights of the model:
- Handles missing and noisy data.
- Performs categorical encoding and scaling.
- Uses advanced feature engineering to create meaningful predictors like PPI and Performance Score.
- Employs a stacking regressor to leverage the strengths of multiple models.

## Data Preprocessing

1) Price Filtering

- Removed extreme outliers in the price column (only considered prices ≤ 150,000).

2) Company Extraction

- Extracted company names from phone model names to create a new company column.

3) Rating Scaling

- Scaled ratings to a 0–10 range for consistency.

4) Feature Removal

- Dropped features with low correlation to price, high missing values, or non-essential attributes.

- Dropped remaining null values since their count was minimal.

5) OS Categorization

- Categorized operating system into Android, iOS, and Other.

6) Chipset Categorization

- Retained chipsets with more than 10 occurrences; all others labeled as Other.

7) Camera and Display Processing

- Renamed camera columns for clarity.

- Split resolution into resx and resy.

## Feature Engineering

- Pixels Per Inch (PPI) – calculated from resolution and display size.

- Performance Score – computed using a combination of RAM, chipset, and other relevant specs.

- Removed highly correlated or redundant features to reduce multicollinearity.

## Modeling

Model: Stacking Regressor

Base Models:

- Random Forest Regressor

- XGBoost Regressor

- LightGBM Regressor

Final Estimator: Ridge Regression

Target Transformation:

Trained the model on the logarithm of price to stabilize variance.
