import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the processed dataset.
    """
    df = pd.read_csv(file_path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Check whether the required columns exist.

    This script assumes the dataset contains:
    - station_id: station identifier
    - hour: hour of day
    - trips: target demand value
    """
    required_cols = ["station_id", "hour", "trips"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if len(missing_cols) > 0:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            "Expected columns: station_id, hour, trips."
        )


def build_station_hour_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a station-level hourly demand profile.

    For each station, compute the average number of trips for each hour.
    The result is a pivot table:

        rows    -> station_id
        columns -> hour_0, hour_1, ..., hour_23

    Missing station-hour combinations are filled with 0.
    """
    profile = (
        df.groupby(["station_id", "hour"])["trips"]
        .mean()
        .reset_index()
        .pivot(index="station_id", columns="hour", values="trips")
        .fillna(0.0)
    )

    # Rename columns to make the output clearer
    profile.columns = [f"hour_{int(col)}" for col in profile.columns]

    return profile


def standardize_profiles(profile_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize station profiles before clustering.

    KMeans is distance-based, so scaling is useful when features
    have different magnitudes.
    """
    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_df)
    return profile_scaled, scaler


def train_kmeans(
    profile_scaled: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42
) -> KMeans:
    """
    Train a KMeans model on the station profiles.

    Parameters:
    - n_clusters: number of station clusters
    - random_state: fixed seed for reproducibility
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    model.fit(profile_scaled)
    return model


def create_station_cluster_table(
    profile_df: pd.DataFrame,
    model: KMeans
) -> pd.DataFrame:
    """
    Create a table mapping each station_id to its cluster label.
    """
    station_clusters = pd.DataFrame({
        "station_id": profile_df.index,
        "station_cluster": model.labels_
    }).reset_index(drop=True)

    return station_clusters


def merge_cluster_labels(
    df: pd.DataFrame,
    station_clusters: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the station cluster labels back into the original dataset.
    """
    merged_df = df.merge(station_clusters, on="station_id", how="left")
    return merged_df


def print_cluster_summary(
    station_clusters: pd.DataFrame,
    profile_df: pd.DataFrame,
    model: KMeans
) -> None:
    """
    Print a short summary of the clustering results.

    This includes:
    - number of stations in each cluster
    - KMeans inertia
    - cluster centers in the scaled feature space
    """
    print("KMeans Clustering Summary")
    print(f"Number of clusters: {model.n_clusters}")
    print(f"Inertia: {model.inertia_:.4f}")

    print("\nStations per cluster:")
    cluster_counts = (
        station_clusters["station_cluster"]
        .value_counts()
        .sort_index()
    )
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} stations")

    centers_df = pd.DataFrame(
        model.cluster_centers_,
        columns=profile_df.columns
    )

    print("\nCluster centers (scaled hourly demand profiles):")
    print(centers_df.round(4))


def save_outputs(
    station_clusters: pd.DataFrame,
    merged_df: pd.DataFrame,
    cluster_file_path: str,
    merged_file_path: str
) -> None:
    """
    Save:
    1. station -> cluster mapping
    2. original dataset with station_cluster added
    """
    station_clusters.to_csv(cluster_file_path, index=False)
    merged_df.to_csv(merged_file_path, index=False)


def main() -> None:
    input_file_path = "data/processed/final_dataset.csv"
    cluster_file_path = "data/processed/station_clusters.csv"
    merged_file_path = "data/processed/final_dataset_with_kmeans.csv"

    # You can change this value if you want to try a different number of clusters
    n_clusters = 5

    # Step 1: Load the processed dataset
    df = load_data(input_file_path)

    # Step 2: Check whether the required columns exist
    validate_columns(df)

    # Step 3: Keep only the columns needed for station clustering
    kmeans_df = df[["station_id", "hour", "trips"]].dropna().copy()

    # Step 4: Build a station-level hourly demand profile
    profile_df = build_station_hour_profile(kmeans_df)

    # Step 5: Standardize the hourly demand profiles
    profile_scaled, _ = standardize_profiles(profile_df)

    # Step 6: Train the KMeans model
    model = train_kmeans(
        profile_scaled=profile_scaled,
        n_clusters=n_clusters,
        random_state=42
    )

    # Step 7: Create a station -> cluster table
    station_clusters = create_station_cluster_table(profile_df, model)

    # Step 8: Merge cluster labels back into the full dataset
    merged_df = merge_cluster_labels(df, station_clusters)

    # Step 9: Print a summary of clustering results
    print_cluster_summary(station_clusters, profile_df, model)

    # Step 10: Save outputs
    save_outputs(
        station_clusters=station_clusters,
        merged_df=merged_df,
        cluster_file_path=cluster_file_path,
        merged_file_path=merged_file_path
    )

    print("\nSaved files:")
    print(f"- {cluster_file_path}")
    print(f"- {merged_file_path}")


if __name__ == "__main__":
    main()