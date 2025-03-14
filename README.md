# House Price Prediction

A beginner machine learning project to predict house prices using the Ames Housing dataset.

## Overview
This project uses a simple linear regression model to predict house prices based on features like living area and overall quality. It’s built with Python and scikit-learn, perfect for learning the basics of AI and data science.

## Features
- **Dataset**: Ames Housing (via `fetch_openml`)
- **Model**: Linear Regression
- **Inputs**: 
  - `GrLivArea` (Above-ground living area in square feet)
  - `OverallQual` (Overall material and finish quality)
- **Output**: Predicted sale price (`PRICE`)

## Requirements
- Python 3.x
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `numpy`

Install them with:
```bash
pip install pandas scikit-learn matplotlib numpy

## How to Run
1. Clone the repo:
```bash
git clone https://github.com/GonzalesMatthew/house-price-prediction.git
cd house-price-prediction
2. Run the script:
```bash
python house_prices.py
3. Check the output:
- Mean Squared Error (MSE) in the terminal.
- A scatter plot of actual vs. predicted prices.
- A predictions.csv file with results.