import json
from pathlib import Path
import pandas as pd
from ..config import (
    COMPETITIONS_PATH,
    MATCHES_ROOT,
    EVENTS_ROOT,
    SHOTS_OUT_PATH,
)

def select_competitions(
    competitions_json_path: Path,
    filter_gender: str | None = None,
) -> pd.DataFrame:
    """
    Reads StatsBomb competitions.json and extracts pairs.

    This function reads a JSON file containing competition data and returns
    a DataFrame with unique (competition_id, season_id) pairs.
    It can optionally filter the results by gender.

    Parameters
    ----------
    competitions_json_path : Path
        Path to the competitions.json file.
    filter_gender : str, optional
        "male" or "female" to filter the results by gender.
        If None (default), no gender filter is applied.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns:
        ['competition_id', 'season_id'].
    """
    # Open the file and load the JSON data
    with competitions_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:
        # Optional gender filter
        if filter_gender is not None and item.get("competition_gender") != filter_gender:
            continue

        # Append row
        rows.append({
            "competition_id": item.get("competition_id"),
            "season_id": item.get("season_id"),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Remove duplicates on (competition_id, season_id)
        df = df.drop_duplicates(subset=["competition_id", "season_id"], keep="last")

        # Sort for readability
        df = df.sort_values(
            by=["competition_id", "season_id"],
            ascending=True
        ).reset_index(drop=True)

    print(df.head())

    return df

def build_matches_index(
    competitions_df: pd.DataFrame,
    matches_root: Path
) -> pd.DataFrame:
    """
    Builds an index of all available matches.

    This function iterates through the provided competitions and seasons,
    reads the match JSON files, and creates a DataFrame that lists
    all available matches.

    Parameters
    ----------
    competitions_df : pd.DataFrame
        A DataFrame containing the `(competition_id, season_id)` pairs
        from which to extract match data.
    matches_root : Path
        The path to the root directory where the match data folders are
        located.

    Returns
    -------
    pd.DataFrame
        A DataFrame with all available matches, containing the columns:
        ['competition_id', 'season_id', 'match_id'].
    """
    rows = []

    # Iterate over unique (competition_id, season_id) pairs
    pairs = (
        competitions_df.loc[:, ["competition_id", "season_id"]]
        .itertuples(index=False)
    )

    for comp_id, season_id in pairs:
        json_path = matches_root / str(comp_id) / f"{season_id}.json"
        if not json_path.exists():
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                matches = json.load(f)
        except Exception as e:
            # Skip if the JSON cannot be read
            continue

        # Each entry in the matches list is one match dictionary
        for m in matches:
            rows.append({
                "competition_id": comp_id,
                "season_id": season_id,
                "match_id": m.get("match_id"),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Deduplicate and sort for readability
        df = df.drop_duplicates(subset=["competition_id", "season_id", "match_id"], keep="last")
        df = df.sort_values(
            by=["competition_id", "season_id", "match_id"],
            ascending=True
        ).reset_index(drop=True)

    return df

def build_shots_dataset(
    matches_df: pd.DataFrame,
    events_root: Path
) -> pd.DataFrame:
    """
    Extracts shot events from all match event files.

    This function reads event data for a list of matches,
    filters for shot events, and extracts specific features
    relevant for building a shot quality model.

    Parameters
    ----------
    matches_df : pd.DataFrame
        A DataFrame with a 'match_id' column listing the matches
        to process. Other columns are ignored.
    events_root : Path
        The path to the root directory where the event JSON files
        are located (e.g., `data/events`).

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a shot. The columns
        include:
        - match_id
        - x, y                         (shot location)
        - play_pattern                 (situational context)
        - shot_type                    (Open Play, Free Kick, Penalty, etc.)
        - shot_body_part               (Right Foot, Left Foot, Head, etc.)
        - shot_technique               (Normal, Volley, etc.)
        - shot_first_time              (bool or None)
        - under_pressure               (bool or None; top-level event field)
        - freeze_frame                 (raw list; to be engineered later)
        - is_goal                      (label: True if outcome == "Goal")
        - shot_statsbomb_xg            (for a final comparison)
    """
    rows = []

    for match_id in matches_df["match_id"].unique():
        json_path = events_root / f"{match_id}.json"
        if not json_path.exists():
            continue

        # Read events JSON safely
        try:
            with json_path.open("r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception:
            continue

        for ev in events:
            # Only keep shots
            if ev.get("type", {}).get("name") != "Shot":
                continue

            shot = ev.get("shot", {}) or {}

            # Label: is_goal
            outcome_name = (shot.get("outcome") or {}).get("name")
            is_goal = True if outcome_name == "Goal" else False

            # Extract context features (player-agnostic)
            row = {
                "match_id": match_id,
                "id": (ev.get("id") or ""),
                # Shot location
                "x": (ev.get("location") or [None, None])[0],
                "y": (ev.get("location") or [None, None])[1],
                # Situational context
                "play_pattern": (ev.get("play_pattern") or {}).get("name"),
                # Shot descriptors
                "shot_type":      (shot.get("type") or {}).get("name"),
                "shot_body_part": (shot.get("body_part") or {}).get("name"),
                "shot_technique": (shot.get("technique") or {}).get("name"),
                "shot_first_time": shot.get("first_time"),
                # Pressure (StatsBomb places this at top-level event)
                "under_pressure": ev.get("under_pressure"),
                # Freeze frame kept raw for later feature engineering (defender/keeper geometry)
                "freeze_frame": shot.get("freeze_frame"),
                # Label
                "is_goal": is_goal,
                # Include the StatsBomb xG value
                "shot_statsbomb_xg": shot.get("statsbomb_xg")
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        # Sort by match_id
        df = df.sort_values(by="match_id", ascending=True).reset_index(drop=True)

    return df

def preprocess_shots(df_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the shots dataset for xG modeling.

    This function performs the following steps:
    1. Drops rows where the ``freeze_frame`` column is missing.
    2. Converts the ``shot_first_time`` and ``under_pressure`` columns
       into boolean flags, where:
         - True = value present
         - False = value missing

    Parameters
    ----------
    df_shots : pandas.DataFrame
        Raw shots dataset, typically as returned by ``build_shots_dataset``.

    Returns
    -------
    pandas.DataFrame
        Cleaned shots dataset with:
        - Only rows where ``freeze_frame`` is available.
        - ``shot_first_time`` and ``under_pressure`` converted to boolean.
    """
    # Keep only rows with freeze_frame available
    df = df_shots.dropna(subset=["freeze_frame"]).copy()

    # Convert 'shot_first_time' and 'under_pressure' to booleans
    df["shot_first_time"] = df["shot_first_time"].notna()
    df["under_pressure"] = df["under_pressure"].notna()

    # 3d) Drop penalty shots
    df = df[df["shot_type"] != "Penalty"].copy()

    return df


def main():
    """
    Builds and saves a shot dataset from StatsBomb Open Data.

    This script orchestrates the data pipeline by:
    1. Selecting male competitions.
    2. Building an index of all available matches.
    3. Extracting and filtering shot events from those matches. Preprocessing the shots df.
    4. Saving the final, cleaned shots dataset to a Parquet file.
    """
    # 1) Competitions
    df_comp = select_competitions(COMPETITIONS_PATH, filter_gender="male")

    # 2) Matches
    df_matches = build_matches_index(df_comp, matches_root=MATCHES_ROOT)

    # 3) Shots
    df_shots = build_shots_dataset(df_matches, events_root=EVENTS_ROOT)

    print("Competitions selected:", len(df_comp))
    print("Matches loaded:", len(df_matches))
    print("Shots extracted:", len(df_shots))
    print(df_shots.head())

    # 3b) Explore missing values and column statistics
    print("\n--- Missing Values per Column ---")
    print(df_shots.isna().sum().sort_values(ascending=False))

    print("\n--- Percentage of Missing Values ---")
    print((df_shots.isna().mean() * 100).round(2).sort_values(ascending=False))

    print("\n--- Dataset Info ---")
    print(df_shots.info())

    print("\n--- Summary Statistics (numeric) ---")
    print(df_shots.describe().T)

    # 3c) Preprocess for xG model
    df_shots = preprocess_shots(df_shots)

    print("\n--- Cleaned Shots ---")
    print(df_shots.info())
    print(df_shots.head())

    # 4) Save as parquet
    SHOTS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_shots.to_parquet(SHOTS_OUT_PATH, index=False)
    df_shots.to_csv("./data/processed/shots.csv", index=False)
    print(f"Shots dataset saved to {SHOTS_OUT_PATH}")


if __name__ == "__main__":
    main()
