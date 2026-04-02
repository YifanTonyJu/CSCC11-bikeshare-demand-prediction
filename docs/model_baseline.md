# Linear Regression Baseline

## 1. Overview

We use Linear Regression as a baseline model for hourly bike demand prediction.  
This model provides a simple and interpretable benchmark for evaluating more advanced models.

---

## 2. Data

We use the processed dataset generated from the raw Toronto Bike Share data.

- Source: 2021–2024 trip-level records
- Aggregation: station-level hourly demand
- Target variable: `trips`

---

## 3. Features

The model uses the following features:

- **Temporal features**: `hour`, `weekday`, `month`, `is_weekend`
- **Station features**: `station_id`, `historical_avg_demand`
- **User feature**: `member_ratio`
- **Lag features**: `lag_1`, `lag_24`

Rows with missing values (caused by lag features) are removed before training.

---

## 4. Data Split

We use a time-based split:

- First 80% of the data → training set  
- Last 20% of the data → test set  

This ensures that future data is not used during training.

---

## 5. Model

We train a standard Linear Regression model using all features without additional encoding or scaling.

This follows the baseline setting defined in the project plan.

---

## 6. Evaluation Metrics

We evaluate the model using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

---

## 7. Results

- MAE: 3.83  
- MSE: 47.27  
- RMSE: 6.88  

---

## 8. Analysis

The Linear Regression model captures general trends in bike demand,  
but its performance is limited due to its linear assumptions.

Bike demand is influenced by complex and nonlinear factors such as time patterns  
and station-specific behavior, which cannot be fully modeled by a linear model.

Therefore, more advanced models such as Random Forest and XGBoost  
are expected to achieve better performance.

---

## 9. Notes

- This model serves as a baseline for comparison.
- No K-means clustering is used at this stage.
- Additional feature encoding and scaling may be explored in later models.