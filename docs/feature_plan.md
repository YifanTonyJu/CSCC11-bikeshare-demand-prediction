# Feature Plan

## Target

**Bike demand** = number of trips starting from a station within a given hour.

---

## Feature Set

### Overview

|   Category    | Features                                            | Description                                                               | Purpose                                           |
| :-----------: | :-------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------ |
| **Temporal**  | `hour`, `weekday`, `month`, `is_weekend`            | Extracted from trip start time to represent temporal patterns.            | Capture daily, weekly, and seasonal trends.       |
|  **Station**  | `station_id`, `historical_avg_demand`               | Station identifier and its average hourly demand over the entire dataset. | Capture station-specific baseline demand.         |
|   **User**    | `member_ratio`                                      | Proportion of trips made by annual members at a given time.               | Reflect user behavior differences.                |
|    **Lag**    | `lag_1`, `lag_24`                                   | Demand in the previous hour and same hour on the previous day.            | Capture short-term and daily temporal dependency. |
| **Extension** | `station cluster label` *(K-means, modeling stage)* | Stations grouped based on demand patterns using clustering.               | Capture similarity between stations.              |

---

## Implementation Notes

- All features are computed using historical data.
- During prediction, some features (e.g., lag features) require recent observations or approximations.
- K-means clustering is not part of preprocessing and will be applied during the modeling stage.