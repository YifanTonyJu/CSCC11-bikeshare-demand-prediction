# CSCC11 Project: Toronto Bike Demand Prediction

Predicting station-level bike demand in Toronto using machine learning.

## Table of Contents

- [CSCC11 Project: Toronto Bike Demand Prediction](#cscc11-project-toronto-bike-demand-prediction)
  - [Table of Contents](#table-of-contents)
  - [Project Goal](#project-goal)
  - [Dataset](#dataset)
  - [Problem Setup](#problem-setup)
  - [Research Question](#research-question)
  - [Feature Plan](#feature-plan)
  - [Model Candidates](#model-candidates)
  - [Evaluation Plan](#evaluation-plan)
  - [Original Contribution](#original-contribution)
  - [Team Roles](#team-roles)
  - [Environment Setup](#environment-setup)
  - [Collaboration Workflow](#collaboration-workflow)

## Project Goal

Build a regression pipeline to predict hourly bike demand per station.

## Dataset

Source: Toronto Bike Share Ridership Data (2021-2024)

The raw data contains trip-level records, including:

- Start time
- Start station
- Trip duration
- Bike ID
- User type

For modeling, data is aggregated by station and hour to form demand labels.

> Note: Raw data files are excluded from version control due to size limits.

## Problem Setup

- Task type: Regression
- Target: Number of trips starting from a station in a given hour

## Research Question

How do different machine learning models compare in predicting station-level bike demand?

## Feature Plan

- Temporal features: hour, day of week, month, weekend
- Station features: station ID, historical average demand
- User behavior feature: proportion of annual members
- Lag features: previous-hour demand, same-hour demand from previous day
- Extension feature: station cluster label from K-means

## Model Candidates

- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost

## Evaluation Plan

- MAE
- MSE
- RMSE
- Time-based train/test split

## Original Contribution

Use K-means to cluster stations by historical demand pattern and use cluster labels as extra model features.

> Note: K-means clustering is planned for the modeling stage.
> The current processed dataset does not yet include cluster labels.

## Team Roles

| Name      | Tasks                                                            | Start Date | End Date   |
| --------- | ---------------------------------------------------------------- | ---------- | ---------- |
| Yifan Ju  | Data preprocessing, feature engineering, dataset construction    | 2026-03-22 | 2026-03-25 |
| Yifan Ju  | Linear Regression baseline and documentation                     | 2026-03-27 | 2026-03-30 |
| Yifan Ju  | Report writing (Data section, partial Introduction, Future Work) | 2026-04-01 | 2026-04-03 |
| Yehan Lin | Ridge regression baseline and mean baseline evaluation           | 2026-03-30 | 2026-04-02 |
| Yehan Lin | Two-layer neural network(if interested, better MSE,poor MAE)     | 2026-04-02 | 2026-04-03 |
| ...       | ...                                                              | ...        | ...        |

## Environment Setup

This project uses uv for environment and dependency management.

```bash
uv sync
```

## Collaboration Workflow

1. Clone the repository.
2. Create a feature branch.
3. Commit regularly with clear messages.
4. Push your branch to GitHub.
5. Open a pull request before merging.

Keep commits focused and avoid direct pushes to main.
