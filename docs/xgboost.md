# Xgboost

## 5. Model

We train an XGBoost regression model for hourly bike demand prediction.  
Before training, boolean features are converted to integers, and any non-numeric features are one-hot encoded so that all model inputs are numeric.  
The model uses `XGBRegressor` with `objective="reg:squarederror"`, `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, and `random_state=42`.  
Because XGBoost is a tree-based model, feature standardization is not required.

## 6. Evaluation Metrics

We evaluate the model using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

These are the same metrics used in the existing baseline documentation, which makes the comparison consistent across models.

---

## 7. Results

- MAE: 2.8016  
- MSE: 25.2606  
- RMSE: 5.0260  

---

## 8. Analysis

The XGBoost model performs better than the Linear Regression baseline.  
In the baseline markdown file, Linear Regression reports MAE = 3.83, MSE = 47.27, and RMSE = 6.88, while the current XGBoost run gives lower values on all three metrics. This suggests that XGBoost is better able to capture nonlinear demand patterns and feature interactions in the bike demand prediction task.

Compared with a simple linear model, XGBoost can model more complex relationships among temporal features, station-related features, user-related features, and lag features.  
This makes it a stronger candidate for the final regression pipeline, especially when bike demand depends on patterns that are not well approximated by a single linear function.

---

## 9. Notes

- This model is an advanced extension of the baseline regression setting.
- Boolean columns are converted to integers before training.
- Non-numeric features are one-hot encoded before fitting the model.
- No feature scaling is used in this implementation.
- Further hyperparameter tuning may improve performance even more.