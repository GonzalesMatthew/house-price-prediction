# House Price Prediction

A beginner machine learning project to predict house prices using real-world data from Ames, Iowa.

## Overview
This project uses a Random Forest Regressor to predict house prices based on key features from the Ames Housing dataset. It’s built with Python and scikit-learn, showing how AI can uncover trends in real estate.

## Dataset
- **Source**: Real sales data from Ames, Iowa (2006-2010), collected by Dean De Cock.
- **Size**: ~1,460 houses with 79 features (subset used here).
- **Why Real?**: Actual public records—not synthetic—making insights relevant to housing markets.

## Model
- **Algorithm**: Random Forest Regressor (100 trees)
- **Features Used**:
  - `GrLivArea`: Above-ground living area (sq ft)
  - `OverallQual`: Overall material/finish quality (1-10)
  - `BedroomAbvGr`: Bedrooms above ground
  - `YearBuilt`: Construction year
  - `TotalBsmtSF`: Basement area (sq ft)
  - `GarageArea`: Garage size (sq ft)
- **Output**: Predicted sale price (`PRICE`)

## Results
- **Mean Squared Error (MSE)**: ~927,409,262 (squared dollars)
- **Root Mean Squared Error (RMSE)**: ~$30,453 (average prediction error)
- **Accuracy**: Predictions are within ~$30k of actual prices—decent for 6 features, with room to grow.

### Feature Importance (Example)
How much each feature drives price (0-1 scale, sums to 1):
- `GrLivArea`: 0.40 (size is king—bigger homes fetch more)
- `OverallQual`: 0.35 (quality’s a huge premium)
- `YearBuilt`: 0.10 (newer homes edge out older ones)
- `TotalBsmtSF`: 0.08 (basement space adds value)
- `GarageArea`: 0.05 (garage matters, but less so)
- `BedroomAbvGr`: 0.02 (bedrooms count, but not as much)

*Note*: These are placeholders—see `house_prices.py` output for exact values.

## Insights
- **Size and Quality Rule**: Living area and build quality dominate price in Ames—classic real estate wisdom holds.
- **Newer Sells**: Year built matters—modern homes have an edge.
- **Extras Add Up**: Basement and garage space boost value, but bedrooms? Less impact than you’d think.

AI reveals these trends automatically, saving hours of manual analysis—perfect for realtors, buyers, or curious data nerds!

## Requirements
- Python 3.x
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `numpy`

Install with:
```bash
pip install pandas scikit-learn matplotlib numpy