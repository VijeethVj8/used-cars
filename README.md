# Used Cars Price Prediction

This project aims to predict the prices of used cars using machine learning techniques. The dataset for this project is sourced from Kaggle's Playground Series (Season 4, Episode 9). The objective is to build a regression model that accurately estimates the price of used cars based on features like brand, model, mileage, transmission type, and more.

## Project Overview

The dataset contains information about various used cars, including features such as:
- `brand`: The car manufacturer
- `model`: The specific model of the car
- `model_year`: Year the car was manufactured
- `milage`: Mileage of the car
- `fuel_type`: Type of fuel used
- `transmission`: Type of transmission (e.g., automatic, manual)
- `ext_col` and `int_col`: External and internal colors of the car
- `accident`: Whether the car has been in an accident
- `clean_title`: Indicates whether the car has a clean title
- `horsepower`: Engine power
- `price`: The target variable to predict

## Approach

1. **Data Preprocessing**
   - Extracted `horsepower` from the `engine` column.
   - Categorical features like `brand`, `model`, `fuel_type`, etc., were encoded using **Target Encoding**.
   - The data was then scaled using **StandardScaler** to bring all features onto a similar scale.

2. **Model Training**
   - Two main models were trained: **CatBoost Regressor** and **LightGBM Regressor**.
   - **CatBoost** was trained with early stopping to prevent overfitting.
   - **LightGBM** was tuned using various parameters, including `learning_rate`, `num_leaves`, `max_depth`, and others.

3. **Evaluation**
   - **Root Mean Square Error (RMSE)** was used as the metric for evaluation.
   - After training, **LightGBM** achieved a slightly better RMSE than CatBoost, indicating it was the better model for this dataset.

4. **Prediction**
   - The best-performing model was used to predict the prices on the test set.
   - Predictions were saved to `test_predictions.csv`.

## Results

- **LightGBM Model** achieved a **Validation RMSE of 67549.41**.
- **CatBoost Model** had a **Validation RMSE of 67838.64**.
- LightGBM performed better in this scenario and was used to generate final predictions.

## Files
- **`train.csv`**: Training data with features and target variable.
- **`test.csv`**: Test data for generating predictions.
- **`usedcars.ipynb`**: Jupyter Notebook containing all the steps from data preprocessing to model evaluation.
- **`test_predictions.csv`**: Output predictions on the test set.
- **`catboost_info/`**: Directory containing training logs for the CatBoost model.

## Requirements
- **Python 3.8+**
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `catboost`
  - `lightgbm`
  - `category_encoders`
  - `matplotlib`

To install the required libraries, run:
```sh
pip install -r requirements.txt
```

## Future Improvements
- **Feature Engineering**: Adding more derived features, such as vehicle age, to capture hidden insights.
- **Hyperparameter Tuning**: Use advanced methods like **Bayesian Optimization** to further tune the models.
- **Stacking and Blending Models**: Combine multiple models to improve overall performance.

## How to Use
1. Clone the repository:
   ```sh
   git clone https://github.com/VijeethVj8/used-cars.git
   ```
2. Install dependencies and run the notebook to train models and generate predictions.

## Acknowledgments
- **Kaggle** for providing the dataset.
- **CatBoost** and **LightGBM** teams for their incredible gradient boosting libraries.

