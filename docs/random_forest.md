# Random Forest Regression

## Model

We train a Random Forest regression model to predict hourly bike demand.

Before training, rows with missing values (mainly caused by lag features) are removed.  
We use a time-based split, where the first 80% of the dataset is used for training and the remaining 20% is used for testing. This ensures that no future information is used during training and avoids data leakage.

The model is implemented using `RandomForestRegressor` with the following parameters:
- n_estimators = 200  
- max_depth = 20  
- min_samples_split = 5  
- min_samples_leaf = 2  
- random_state = 42  

All features are used in their original numeric form. No feature scaling or encoding is applied, as Random Forest is a tree-based model and is invariant to feature scaling.

---

## Evaluation Metrics

We evaluate the model using the following metrics:

- MAE (Mean Absolute Error)  
- MSE (Mean Squared Error)  
- RMSE (Root Mean Squared Error)  

These metrics allow direct comparison with baseline models such as Linear Regression and Ridge Regression.

---

## Results

- MAE: 2.7555  
- MSE: 26.4067  
- RMSE: 5.1387  

---

## Analysis

The Random Forest model significantly outperforms the baseline linear models.  
Compared to Linear Regression and Ridge Regression, Random Forest achieves lower error (RMSE) and higher R², indicating better predictive performance.

When compared with XGBoost, Random Forest shows slightly worse performance.  
XGBoost achieves RMSE = 5.0260, which is lower than the Random Forest RMSE of 5.1387, indicating that XGBoost provides the most accurate predictions among all models.

This improvement is mainly due to the ability of tree-based ensemble models to capture nonlinear relationships and complex feature interactions.  
Bike demand depends on temporal patterns (such as hour of the day and day of the week) and strong short-term dependencies, which are not well modeled by linear approaches.

Feature importance analysis shows that lag-based features, especially `lag_1`, dominate the model, indicating that recent demand is the most important predictor of current demand.  
Other features such as `hour` and `lag_24` also contribute, reflecting daily patterns in bike usage.

Although Random Forest performs very well without additional preprocessing, XGBoost further improves performance by using gradient boosting, which sequentially corrects errors from previous trees and provides better optimization.

Overall, Random Forest offers strong performance with a simple and robust structure, while XGBoost achieves the best predictive accuracy among all models.

---

## Notes

- Random Forest is used as a nonlinear extension of baseline models.  
- No feature scaling is applied.  
- No encoding is required since all features are numeric.  
- The model is robust and performs well with minimal preprocessing.  
- Strong performance is mainly driven by lag features capturing temporal dependencies.  