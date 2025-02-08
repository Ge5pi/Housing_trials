# Ames Housing Price Prediction

## Overview
This project uses the Ames Housing dataset to predict house prices based on various features. The dataset includes detailed information about houses, such as their size, quality, and location. The script preprocesses the data, selects important features, and trains machine learning models to predict the target variable, `SalePrice`.

## Files
- **data_description.txt**: Provides detailed descriptions of all the features in the dataset.
- **train.csv**: The dataset used for training and testing the models.
- **main.py**: The main Python script that preprocesses the data, trains models, and evaluates their performance.

## Features Used
The script uses a combination of numerical and categorical features, including:
- `MSSubClass`: Type of dwelling involved in the sale.
- `MSZoning`: General zoning classification of the sale.
- `LotFrontage`: Linear feet of street connected to the property.
- `OverallQual`: Rates the overall material and finish of the house.
- `YearBuilt`: Original construction date.
- `GrLivArea`: Above grade (ground) living area square feet.
- And many more (see `main.py` for the full list).
