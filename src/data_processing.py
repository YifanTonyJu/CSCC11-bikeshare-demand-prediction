import os
import pandas as pd


def load_data(base_path: str) -> pd.DataFrame:
    """
    Load and combine all CSV files under data/raw/.
    """
    dfs = []

    for year_folder in sorted(os.listdir(base_path)):
        year_path = os.path.join(base_path, year_folder)

        if os.path.isdir(year_path):
            for file in sorted(os.listdir(year_path)):
                if file.endswith(".csv"):
                    file_path = os.path.join(year_path, file)
                    print(f"Loading: {file_path}")
                    df_temp = pd.read_csv(file_path, encoding="latin1")
                    dfs.append(df_temp)

    if not dfs:
        raise ValueError("No CSV files found in the given directory.")

    df = pd.concat(dfs, ignore_index=True)
    print("Data loaded successfully.")
    print(f"Total rows: {len(df)}")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names, remove unwanted columns, keep needed columns,
    and rename them to simpler names.
    """
    # Step 1: clean column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )

    # Step 2: remove unwanted extra column caused by encoding issues
    df = df.drop(columns=["ï»¿Trip_Id"], errors="ignore")

    # Step 3: keep only the columns we need
    cols_needed = [
        "Start_Station_Id",
        "Start_Time",
        "User_Type",
    ]
    df = df[cols_needed].copy()

    # Step 4: drop rows with missing key values
    df = df.dropna(subset=["Start_Station_Id", "Start_Time", "User_Type"])

    # Step 5: rename columns
    df = df.rename(columns={
        "Start_Station_Id": "station_id",
        "Start_Time": "start_time",
        "User_Type": "user_type",
    })

    print("Preprocessing completed.")
    print(df.head())

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal and user-related features.
    """
    df["start_time"] = pd.to_datetime(df["start_time"])

    df["hour"] = df["start_time"].dt.hour
    df["weekday"] = df["start_time"].dt.weekday
    df["month"] = df["start_time"].dt.month
    df["is_weekend"] = df["weekday"] >= 5

    df["is_member"] = df["user_type"] == "Annual Member"

    print("Feature engineering completed.")
    print(df.head())

    return df


def construct_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trips and construct demand-related features.
    """
    # Construct target: trips
    df_demand = (
        df.groupby(["station_id", "hour", "weekday", "month"])
        .size()
        .reset_index(name="trips")
    )

    # Sort before creating lag features
    df_demand = df_demand.sort_values(
        by=["station_id", "month", "weekday", "hour"]
    ).reset_index(drop=True)

    # Lag features
    df_demand["lag_1"] = df_demand.groupby("station_id")["trips"].shift(1)
    df_demand["lag_24"] = df_demand.groupby("station_id")["trips"].shift(24)

    # User feature: proportion of annual members
    df_user = (
        df.groupby(["station_id", "hour", "weekday", "month"])["is_member"]
        .mean()
        .reset_index(name="member_ratio")
    )

    df_demand = df_demand.merge(
        df_user,
        on=["station_id", "hour", "weekday", "month"],
        how="left",
    )

    # Weekend indicator in final table
    df_demand["is_weekend"] = df_demand["weekday"] >= 5

    # Historical average demand for each station
    df_station_avg = (
        df_demand.groupby("station_id")["trips"]
        .mean()
        .reset_index(name="historical_avg_demand")
    )

    df_demand = df_demand.merge(
        df_station_avg,
        on="station_id",
        how="left",
    )

    # Move target column to the end
    cols = [c for c in df_demand.columns if c != "trips"] + ["trips"]
    df_demand = df_demand[cols]

    print("Demand construction completed.")
    print(df_demand.head())

    return df_demand

def main() -> None:
    base_path = "data/raw"
    output_path = "data/processed/final_dataset.csv"

    os.makedirs("data/processed", exist_ok=True)

    df = load_data(base_path)
    df = preprocess(df)
    df = feature_engineering(df)
    df_final = construct_demand(df)

    df_final.to_csv(output_path, index=False)
    print(f"Final dataset saved to: {output_path}")


if __name__ == "__main__":
    main()